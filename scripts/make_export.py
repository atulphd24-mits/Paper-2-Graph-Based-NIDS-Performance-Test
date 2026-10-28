"""
Build paper2_export/ (results summary, figures, result CSVs, final logs) and zip it.

Every number in results_summary.md is computed here from the result CSVs, so the
summary always matches the files shipped next to it.

Usage:
    python scripts/make_export.py
"""

import json
import platform
import shutil
import subprocess
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LS = ROOT / "outputs/reports/leakage_study"
EXT = ROOT / "outputs/reports/external"
FIG = ROOT / "outputs/figures/paper"
OUT = ROOT / "paper2_export"

METRICS = [
    ("TPR@FPR1pct", "TPR@1%FPR"),
    ("TPR@FPR0.1pct", "TPR@0.1%FPR"),
    ("testFPR@1pct", "FPR at 1% thr."),
    ("Precision_w", "Precision (w)"),
    ("F1_w", "F1 (w)"),
    ("PR_AUC_w", "PR-AUC (w)"),
    ("ROC_AUC", "ROC-AUC"),
]

FINAL_LOGS = {
    LS: ["run_full_pipeline.log", "run_label_ambiguity.log", "run_build_sample.log",
         "run_part1.log", "run_part34.log", "run_part3_blockfix.log", "run_part5.log",
         "run_robustA.log", "run_robustB.log", "run_egraphsage.log", "run_figures.log",
         "run_paper_tables.log"],
    EXT: ["run_unsw.log", "run_ciciot.log"],
}


# ----------------------------------------------------------------------
# Markdown helpers
# ----------------------------------------------------------------------

def md(df, floatfmt=4):
    """DataFrame -> GitHub markdown table."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "–" if pd.isna(v) else f"{v:.{floatfmt}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def mean_std(raw, keys, metrics=METRICS, digits=4):
    """Group raw per-seed rows and format every metric as 'mean ± std (n)'."""
    cols = [c for c, _ in metrics if c in raw.columns]
    g = raw.groupby(keys, sort=False)
    out = g.size().rename("seeds").reset_index()
    for c, label in metrics:
        if c not in raw.columns:
            continue
        m, s = g[c].mean().values, g[c].std().values
        out[label] = [f"{a:.{digits}f} ± {b:.{digits}f}" if not pd.isna(b) else f"{a:.{digits}f}"
                      for a, b in zip(m, s)]
    return out


def sig_table(path, metric="TPR@FPR1pct"):
    t = pd.read_csv(path)
    t = t[t.metric == metric].drop(columns=["metric"])
    return t.rename(columns={"mean_diff": "mean diff vs baseline", "wins": "seeds better",
                             "wilcoxon_p": "Wilcoxon p"})


def read(path):
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

def environment():
    cpu = subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | sed 's/.*: *//'"],
                         capture_output=True, text=True).stdout.strip()
    import sklearn, torch, torch_geometric, scipy  # noqa: E401
    return {
        "CPU": f"{cpu} (24 logical cores)",
        "Memory": "62 GB RAM",
        "GPU": "none (all models trained on CPU)",
        "OS": f"{platform.system()} {platform.release()}",
        "Python": platform.python_version(),
        "scikit-learn": sklearn.__version__, "PyTorch": torch.__version__,
        "PyTorch Geometric": torch_geometric.__version__, "pandas": pd.__version__,
        "NumPy": np.__version__, "SciPy": scipy.__version__,
    }


def pipeline_times():
    lines = (LS / "run_full_pipeline.log").read_text().splitlines()
    stamps = [(pd.Timestamp(l[4:23]), l[24:].strip()) for l in lines if l.startswith("===")]
    rows = [{"stage": a[1], "start": str(a[0]), "wall-clock": str(b[0] - a[0])}
            for a, b in zip(stamps, stamps[1:]) if a[1] != "done"]
    return pd.DataFrame(rows)


def class_tables():
    fam = read(LS / "label_ambiguity_by_family.csv")
    src = read(LS / "label_ambiguity_by_source.csv")
    rows = []
    for r in src.itertuples():
        rows.append({"source": r.source, "class": "Benign",
                     "flows": int(round(r.flows * (1 - r.attack_share)))})
    for r in fam.itertuples():
        rows.append({"source": r.source, "class": r.attack, "flows": int(r.flows)})
    bigflow = pd.DataFrame(rows).sort_values(["source", "flows"], ascending=[True, False])
    bigflow["share of source (%)"] = bigflow.groupby("source")["flows"].transform(
        lambda x: 100 * x / x.sum()).round(3)
    external = []
    for tag, name in (("unsw", "UNSW-NB15 (original)"), ("ciciot", "CIC-IoT2023")):
        counts = json.loads((EXT / f"{tag}_class_counts.json").read_text())
        total = sum(counts.values())
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            external.append({"dataset": name, "class": k, "flows": v,
                             "share (%)": round(100 * v / total, 3)})
    return bigflow, pd.DataFrame(external)


def working_samples():
    rows = []
    bf = pd.read_parquet(ROOT / "data/processed/bigflow_working_sample.parquet", columns=["source", "label"])
    for s, g in bf.groupby("source"):
        rows.append({"dataset": f"BigFlow: {s}", "sample flows": len(g), "attack share": g.label.mean()})
    for tag, name in (("unsw", "UNSW-NB15 (original)"), ("ciciot", "CIC-IoT2023")):
        d = pd.read_parquet(ROOT / f"data/processed/{tag}_working_sample.parquet", columns=["label"])
        rows.append({"dataset": name, "sample flows": len(d), "attack share": d.label.mean()})
    return pd.DataFrame(rows)


def build_summary():
    env = environment()
    bigflow_classes, external_classes = class_tables()
    S = []
    add = S.append

    add(f"# Results summary — leakage audit of graph-based NIDS\n")
    add(f"Generated {date.today().isoformat()} by `scripts/make_export.py` from the result CSVs in this "
        "folder. Every number below is computed from those files.\n")
    add("**Study:** *How Much of Graph-Based Intrusion Detection Performance Is Real? Shortcut, Split "
        "and Label Leakage in a Merged NetFlow Benchmark.* An evaluation study of BigFlow-NIDS "
        "(66.9M flows), with external validation on the original UNSW-NB15 and on CIC-IoT2023.\n")
    add("## 0. Scope notes (read first)\n")
    add("- **No \"adaptive filtering method\" exists in this study.** The closest component is the "
        "*family-gated detector* (§2.6), which adaptively chooses a feature-set expert per attack "
        "family. It is described in full below.\n"
        "- **Detection/response time, latency and throughput were not measured.** No inference-latency "
        "or throughput experiment was run. §5.12 reports only the wall-clock time of each experiment "
        "stage, which is not a detection-latency measure.\n"
        "- **Accuracy** is reported only for the pooled shortcut analysis (§5.1). The main experiments "
        "use operational metrics instead: detection rate (TPR) at a fixed 1% / 0.1% false-positive "
        "rate, the false-positive rate actually reached on test data, prevalence-weighted precision, "
        "F1 and PR-AUC, and ROC-AUC. Accuracy is uninformative at these class imbalances "
        "(e.g. 97.7% attacks in CIC-IoT2023).\n"
        "- `(w)` = weighted so that test flows have each source's real class prevalence (§1.4).\n"
        "- Values are mean ± sample standard deviation over seeds; `seeds` gives the number of runs.\n")

    # ---------------------------------------------------------------- 1. datasets
    add("## 1. Datasets\n")
    add("### 1.1 Overview\n")
    src = read(LS / "label_ambiguity_by_source.csv")
    unsw_s = read(EXT / "unsw_ambiguity_summary.csv").iloc[0]
    cic_s = read(EXT / "ciciot_ambiguity_summary.csv").iloc[0]
    ov = pd.DataFrame([
        {"dataset": "BigFlow-NIDS (Mendeley v2, DOI 10.17632/nv729tbdgz.2)", "rows": int(src.flows.sum()),
         "features used": 47, "attack share": float(np.average(src.attack_share, weights=src.flows)),
         "timestamps / IPs": "yes / yes", "notes": "merge of NetFlow-v3 UNSW-NB15, CSE-CIC-IDS2018, BoT-IoT, ToN-IoT; 67 files"},
        {"dataset": "UNSW-NB15 (original, 4 raw CSVs)", "rows": int(unsw_s.rows), "features used": 41,
         "attack share": float(unsw_s.attack_share), "timestamps / IPs": "yes / yes",
         "notes": "Argus/Bro features; plus official train/test partition"},
        {"dataset": "CIC-IoT2023 (feature CSVs, 169 files)", "rows": int(cic_s.rows), "features used": 46,
         "attack share": float(cic_s.attack_share), "timestamps / IPs": "no / no",
         "notes": "packet-window features, 34 labels; partial protocol only"},
    ])
    add(md(ov) + "\n")
    add("BigFlow by recovered source dataset:\n")
    add(md(src.rename(columns={"attack_flows_in_ambiguous_vectors": "attack flows in ambiguous vectors"})) + "\n")

    add("### 1.2 Class distribution (all rows)\n")
    add("BigFlow-NIDS, per source and class:\n")
    add(md(bigflow_classes, floatfmt=3) + "\n")
    add("External datasets:\n")
    add(md(external_classes, floatfmt=3) + "\n")
    tr = pd.read_csv(ROOT / "data/raw/UNSW-NB15/UNSW_NB15_training-set.csv", usecols=["attack_cat"])
    te = pd.read_csv(ROOT / "data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv", usecols=["attack_cat"])
    off = pd.concat([tr.attack_cat.value_counts().rename("train"), te.attack_cat.value_counts().rename("test")], axis=1)
    off.index.name = "class"
    add("UNSW-NB15 official partition (175,341 train / 82,332 test):\n")
    add(md(off.reset_index()) + "\n")

    add("### 1.3 Preprocessing\n")
    add("- **Source recovery (BigFlow):** each flow is assigned to its source dataset from its start "
        "timestamp (UNSW-NB15 < 2016; CSE-CIC-IDS2018 Jan–mid-Apr 2018; BoT-IoT mid-Apr–Dec 2018; "
        "ToN-IoT 2019); checked against testbed address ranges on 1.2M flows from 12 files.\n"
        "- **Behaviour features:** removed flow start/end timestamps, source and destination IP, "
        "source port and DNS query id (BigFlow: 47 features remain). UNSW-NB15: removed srcip, sport, "
        "dstip, dsport, stime, ltime (41 remain; `proto`, `state`, `service` integer-coded). "
        "CIC-IoT2023: all 46 features (no identifiers present).\n"
        "- **Distance-based inputs** (neighbourhood features): log1p of heavy-tailed non-negative "
        "counters (skew > 2), one-hot of destination port / protocol / L7 protocol (top 20 + other; "
        "BigFlow only), TCP flag bits, standard scaling — all fitted on training rows only.\n"
        "- **MLP inputs:** sign(x)·log(1+|x|), then standard scaling (fit on training rows).\n"
        "- **Working samples:** capped at 20,000 flows per (source, attack class) and 100,000 benign "
        "per source (all rows kept for rarer classes). Context experiments subsample each source to "
        "60,000 flows stratified by class.\n")
    add(md(working_samples()) + "\n")

    add("### 1.4 Train / validation / test splits\n")
    add("All splits are 70% train / 15% validation / 15% test, per source dataset.\n\n"
        "- **Random rows:** stratified by attack class (the common practice).\n"
        "- **10-minute time blocks:** whole 10-min windows assigned at random to a split, then "
        "repaired so every class with ≥ 3 blocks is in all splits.\n"
        "- **Attack episodes (proposed protocol):** a new episode starts after > 5 min without a flow "
        "of that class; benign flows grouped into 30-min blocks; whole episodes assigned to splits. "
        "Classes with < 3 episodes fall back to a chronological split (earliest 70% train, latest "
        "15% test) — this affects 20% of ToN-IoT and 33% of CSE-CIC-IDS2018 test flows.\n"
        "- **UNSW-NB15 official partition:** the published training/testing files; 15% of the "
        "training file held out as validation.\n"
        "- CIC-IoT2023: random rows only (no timestamps).\n")

    add("### 1.5 Leakage-audit steps\n")
    add("1. Source-identifiability test: classifier predicting the source dataset from behaviour "
        "features; single-feature tests.\n"
        "2. Exact-duplicate analysis on **all** rows: hash of each behaviour vector; share of attack "
        "flows whose vector also occurs as benign; irreducible error = Σ_v min(benign_v, attack_v) / N.\n"
        "3. Single-feature shortcut test (external datasets): balanced accuracy of a model on one "
        "feature.\n"
        "4. Timestamp-only control model under every split strategy.\n"
        "5. Train/test duplicate overlap: share of test rows whose exact vector occurs in training.\n"
        "6. Operational metrics: alert threshold = (1 − FPR) quantile of **validation** benign "
        "scores, applied unchanged to test; the FPR actually reached on test is reported.\n"
        "7. Prevalence weighting: test (and validation, for the F1 threshold) flows weighted by "
        "w = N_{source,class} / n_{source,class} (full-data count / sample count).\n"
        "8. Neighbourhood label features use training labels only, with each training flow's own "
        "label excluded (leave-one-out).\n")

    # ---------------------------------------------------------------- 2. methods
    add("## 2. Models and methods\n")
    add("### 2.1 Gradient boosting (HistGradientBoostingClassifier)\n")
    add("300 iterations, learning rate 0.1, `class_weight='balanced'`, scikit-learn defaults otherwise "
        "(early stopping 'auto').\n")
    add("### 2.2 Random forest\n")
    add("200 trees, `min_samples_leaf=2`, `max_features='sqrt'`, `class_weight='balanced_subsample'`.\n")
    add("### 2.3 MLP\n")
    add("Hidden layers (128, 64), ReLU, Adam (lr 1e-3), L2 alpha 1e-4, batch 512, ≤ 100 epochs, early "
        "stopping (10% internal validation, patience 8); minority class oversampled to balance; inputs "
        "signed-log + standardised.\n")
    add("### 2.4 Context features\n")
    add("- **Graph-neighbourhood (k-NN, k = 20)** in the transformed feature space. Label-aware: share "
        "of attack neighbours, distance-weighted share, nearest-neighbour label, per-family shares "
        "(training labels only, leave-one-out). Label-free: distance to 1st and k-th neighbour, mean "
        "neighbour distance, local outlier factor, neighbour mean / std / deviation of six "
        "volume and TTL columns.\n"
        "- **Host behaviour (78 features, BigFlow and UNSW-NB15):** for each flow, statistics of its "
        "source host, destination host and source–destination pair within 1 s, 10 s and 60 s "
        "windows — flow count, distinct peers, distinct destination ports, destination-port entropy "
        "and ratio, SYN-without-ACK share, unanswered share, tiny-flow (≤ 2 packets) share, mean "
        "log bytes. Computed over all flows of the dataset; no labels used.\n")
    add("### 2.5 E-GraphSAGE (re-implemented)\n")
    add("Flows are directed edges between endpoints carrying the transformed flow features (signed "
        "log + standard scaling, fit on train). Node states start as constant 1-vectors. Two layers: "
        "message = Linear([h_u ‖ x_e]), mean-aggregated over incident edges in both directions, "
        "update h_v = ReLU(Linear([h_v ‖ m_v])), hidden 128, dropout 0.2. Each flow is classified "
        "from [z_u ‖ z_v]. Full-batch transductive training (all flows in the graph, loss on training "
        "flows), Adam lr 1e-3, weight decay 1e-5, class-weighted cross-entropy, ≤ 300 epochs, early "
        "stopping on validation PR-AUC (checked every 5 epochs, patience 30), 8 CPU threads. "
        "Endpoint definitions: **IP**, **IP:port**, **random** (each flow gets two private nodes — "
        "control with no shared structure). Re-implemented from Lo et al. (NOMS 2022); not the "
        "authors' code.\n")
    add("### 2.6 Family-gated detector (the adaptive component)\n")
    add("Motivation: whether context helps differs between datasets (helps CSE-CIC-IDS2018, hurts "
        "ToN-IoT) and is unknown in advance.\n\n"
        "1. **Experts:** three gradient-boosting detectors on flow, flow + neighbourhood and "
        "flow + host features.\n"
        "2. **Router:** a multiclass gradient-boosting model (200 iterations, balanced class weights, "
        "no early stopping) predicting the attack family (incl. Benign) from flow features.\n"
        "3. **Per-family expert choice (validation only):** for each family, compute each expert's "
        "TPR at 1% FPR on validation flows of that family; choose the best, preferring the flow "
        "expert on ties; families absent from validation default to the flow expert.\n"
        "4. **Calibration:** each expert's score is mapped to the fraction of validation benign "
        "scores it exceeds (+1e-3 × raw score to break ties), so experts share one scale.\n"
        "5. **Combination:** final score = Σ_f p(f | x) · score_{expert(f)}(x), where p(f | x) is the "
        "router's probability for attack family f, renormalised over attack families.\n"
        "6. Baseline for the adaptivity: the unweighted mean of the three calibrated experts.\n")

    # ---------------------------------------------------------------- 3. hyperparameters
    add("## 3. Hyperparameters and training setup\n")
    hp = pd.DataFrame([
        ("Working-sample caps", "20,000 per attack class, 100,000 benign per source"),
        ("Context-experiment size", "60,000 flows per source (stratified)"),
        ("Split proportions", "70 / 15 / 15"),
        ("Episode gap / benign block", "5 min (1 and 30 min in sensitivity) / 30 min"),
        ("Time block (block split)", "10 min"),
        ("Min episodes before chronological fallback", "3"),
        ("k (neighbourhood)", "20"),
        ("Host windows", "1 s, 10 s, 60 s"),
        ("Target FPRs", "1% and 0.1%"),
        ("Seeds", "10 (context, E-GraphSAGE, gated, gap); 5 (split comparison); 3 (pooled shortcuts)"),
        ("Statistical test", "two-sided paired Wilcoxon signed-rank over seeds (min p = 0.002 with 10)"),
        ("HistGB", "300 iter, lr 0.1, balanced weights"),
        ("Random forest", "200 trees, min_samples_leaf 2, sqrt features, balanced_subsample"),
        ("MLP", "(128, 64), Adam 1e-3, alpha 1e-4, batch 512, ≤ 100 epochs, early stopping"),
        ("E-GraphSAGE", "2 layers, hidden 128, dropout 0.2, Adam 1e-3, wd 1e-5, ≤ 300 epochs, patience 30"),
        ("Gated router", "HistGB multiclass, 200 iter, no early stopping"),
    ], columns=["setting", "value"])
    add(md(hp) + "\n")
    add("**Environment:**\n")
    add(md(pd.DataFrame(list(env.items()), columns=["item", "value"])) + "\n")

    # ---------------------------------------------------------------- 4. baselines
    add("## 4. Baselines compared\n")
    add("- Flow-only gradient boosting, random forest and MLP (reference for every context comparison).\n"
        "- Random-row split (reference for leakage comparisons) and 10-minute time-block split.\n"
        "- Timestamp-only and source-dataset-only models (shortcut controls).\n"
        "- E-GraphSAGE (published graph IDS) with IP, IP:port and random-node endpoints.\n"
        "- Unweighted expert ensemble (baseline for the family-gated detector).\n"
        "- UNSW-NB15 official training/testing partition.\n")

    # ---------------------------------------------------------------- 5. results
    add("## 5. Results\n")
    p1 = read(LS / "part1_shortcuts_raw.csv")
    det = p1[p1.task == "pooled attack detection"]
    add("### 5.1 Shortcuts on pooled BigFlow (random split, 3 seeds)\n")
    add(md(mean_std(det, ["Model"], [(c, c) for c in
            ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR", "ROC_AUC", "PR_AUC"]])) + "\n")
    sid = p1[p1.task == "identify source from behaviour"]["balanced_accuracy"]
    add(f"Source identification from behaviour features: balanced accuracy {sid.mean():.4f} ± {sid.std():.4f}.\n")
    add("Best single features for source identification:\n")
    add(md(read(LS / "part1_single_feature_source_id.csv").head(10)) + "\n")

    add("### 5.2 Label ambiguity per attack family (all rows)\n")
    add(md(read(LS / "label_ambiguity_by_family.csv")) + "\n")

    add("### 5.3 Split strategies (gradient boosting, 5 seeds)\n")
    p3 = read(LS / "part3_splits_raw.csv")
    t = mean_std(p3, ["source", "representation", "split"])
    dup = p3.groupby(["source", "representation", "split"], sort=False)["dup_overlap"].mean().values
    t["test rows duplicated in train"] = dup
    add(md(t) + "\n")

    add("### 5.4 Split inflation across model families (5 seeds; E-GraphSAGE 10)\n")
    a1 = read(LS / "robustA1_splits_models_raw.csv")
    add(md(mean_std(a1, ["source", "model", "split"])) + "\n")

    add("### 5.5 Context features (attack-episode split, gradient boosting, 10 seeds)\n")
    p4 = read(LS / "part4_graph_raw.csv")
    add(md(mean_std(p4, ["source", "representation"])) + "\n")
    add("Per-family TPR at 1% FPR (mean over 10 seeds):\n")
    fam_cols = [c for c in p4.columns if c.startswith("TPR@FPR1pct_")]
    for s in ["ToN-IoT", "CSE-CIC-IDS2018"]:
        g = p4[p4.source == s].groupby("representation")[fam_cols].mean().T.dropna(how="all")
        g.index = [i.replace("TPR@FPR1pct_", "") for i in g.index]
        g.index.name = f"{s} family"
        add(md(g.reset_index()) + "\n")
    add("Significance vs flow features:\n")
    add(md(sig_table(LS / "part4_graph_significance.csv")) + "\n")

    add("### 5.6 Context features across model families (10 seeds)\n")
    a2 = read(LS / "robustA2_features_models_raw.csv")
    add(md(mean_std(a2, ["source", "model", "representation"])) + "\n")
    add(md(sig_table(LS / "robustA2_features_models_significance.csv")) + "\n")

    add("### 5.7 E-GraphSAGE (10 seeds)\n")
    eg = read(LS / "egraphsage_raw.csv")
    add(md(mean_std(eg, ["source", "split", "model"])) + "\n")
    add(md(sig_table(LS / "egraphsage_significance.csv")) + "\n")

    add("### 5.8 Family-gated detector (10 seeds)\n")
    p5 = read(LS / "part5_two_branch_raw.csv")
    add(md(mean_std(p5, ["source", "representation"])) + "\n")
    add(md(sig_table(LS / "part5_two_branch_significance.csv")) + "\n")
    add("Expert chosen per family (count over seeds):\n")
    add(md(read(LS / "part5_expert_choices.csv")) + "\n")

    add("### 5.9 Episode-gap sensitivity (gradient boosting, 10 seeds)\n")
    gb = read(LS / "robustB_gap_raw.csv")
    add(md(mean_std(gb, ["source", "gap_min", "representation"])) + "\n")
    add(md(sig_table(LS / "robustB_gap_significance.csv")) + "\n")
    add("Episodes per family by gap (minutes):\n")
    add(md(read(LS / "robustB_gap_episode_counts.csv")) + "\n")

    add("### 5.10 External validation — UNSW-NB15 (original)\n")
    add(md(read(EXT / "unsw_ambiguity_summary.csv")) + "\n")
    add(md(read(EXT / "unsw_ambiguity_by_family.csv")) + "\n")
    add("Single-feature shortcuts (top 10):\n")
    add(md(read(EXT / "unsw_single_feature.csv").head(10)) + "\n")
    us = read(EXT / "unsw_splits_raw.csv")
    t = mean_std(us, ["representation", "split"])
    t["test rows duplicated in train"] = us.groupby(["representation", "split"], sort=False)["dup_overlap"].mean().values
    add(md(t) + "\n")
    off = read(EXT / "unsw_official_split_raw.csv").assign(split="official partition")
    t = mean_std(off, ["split"])
    t["test rows duplicated in train"] = off["dup_overlap"].mean()
    add("Official training/testing partition (gradient boosting):\n")
    add(md(t) + "\n")
    uc = read(EXT / "unsw_context_raw.csv")
    add("Context features and E-GraphSAGE (attack-episode split, 10 seeds):\n")
    add(md(mean_std(uc, ["model", "representation"])) + "\n")
    add(md(sig_table(EXT / "unsw_context_significance.csv")) + "\n")

    add("### 5.11 External validation — CIC-IoT2023\n")
    add(md(read(EXT / "ciciot_ambiguity_summary.csv")) + "\n")
    add("Single-feature shortcuts (top 10):\n")
    add(md(read(EXT / "ciciot_single_feature.csv").head(10)) + "\n")
    cs = read(EXT / "ciciot_splits_raw.csv")
    t = mean_std(cs, ["representation", "split"])
    t["test rows duplicated in train"] = cs.groupby(["representation", "split"], sort=False)["dup_overlap"].mean().values
    add(md(t) + "\n")
    cc = read(EXT / "ciciot_context_raw.csv")
    add("Context features (random split, 10 seeds):\n")
    add(md(mean_std(cc, ["model", "representation"])) + "\n")
    add(md(sig_table(EXT / "ciciot_context_significance.csv")) + "\n")

    add("### 5.12 Experiment wall-clock times (BigFlow pipeline)\n")
    add("Not a detection-latency measure; end-to-end time of each experiment stage on the hardware in §3.\n")
    add(md(pipeline_times()) + "\n")

    # ---------------------------------------------------------------- 6. ablations
    add("## 6. Ablations (where to find them)\n")
    add("- Context feature sets (flow / + neighbourhood / + host / + both): §5.5, §5.6, §5.10, §5.11.\n"
        "- Graph structure (E-GraphSAGE IP vs IP:port vs random-node control): §5.7, §5.10.\n"
        "- Adaptive routing vs plain ensemble (family-gated detector vs expert mean): §5.8.\n"
        "- Split strategy: §5.3, §5.4, §5.10. Episode gap: §5.9.\n"
        "- Structural centralities (degree, PageRank, component size) were dropped after an early "
        "3-seed run on a 46M-flow subset showed them unstable; that run is not part of this export.\n")

    add("## 7. Statistical testing\n")
    add("All model comparisons: paired two-sided Wilcoxon signed-rank test over seeds (each seed = one "
        "split + one model fit). With 10 seeds the smallest attainable p is 0.002; with 5 it is "
        "0.0625, so 5-seed comparisons (split strategies) are reported descriptively only. Tables "
        "above give `seeds better` (wins) and p for TPR@1%FPR; the CSVs in `results/` also contain "
        "the PR-AUC tests.\n")

    add("## 8. Known limitations\n")
    add("- Chronological fallback for classes with < 3 episodes (20% of ToN-IoT, 33% of CIC-IDS2018 "
        "test flows): the episode split is leakage-reduced, not leakage-free.\n"
        "- Only ToN-IoT and CSE-CIC-IDS2018 support a meaningful benign-vs-attack comparison within "
        "BigFlow (BoT-IoT is 99.7% attacks; UNSW-NB15 saturates).\n"
        "- CIC-IoT2023 has no timestamps or addresses (partial protocol), and its continuous features "
        "make exact-match ambiguity uninformative.\n"
        "- E-GraphSAGE is a re-implementation.\n"
        "- Latency, throughput and detection/response time were not measured.\n")

    add("## 9. Folder contents\n")
    add("- `figures/` — all paper figures (PDF + PNG).\n"
        "- `results/bigflow/` — BigFlow result CSVs (raw per-seed, summaries, significance).\n"
        "- `results/external/` — UNSW-NB15 and CIC-IoT2023 result CSVs and class counts.\n"
        "- `results/paper_tables/` — the LaTeX tables of the paper and their CSVs.\n"
        "- `logs/` — output logs of the final runs (see `logs/README.md`).\n")
    return "\n".join(S)


def copy_files():
    for sub in ("figures", "results/bigflow", "results/external", "results/paper_tables", "logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for f in FIG.glob("*"):
        if f.suffix in (".pdf", ".png"):
            shutil.copy2(f, OUT / "figures" / f.name)
    for f in LS.glob("*.csv"):
        shutil.copy2(f, OUT / "results/bigflow" / f.name)
    for f in list(EXT.glob("*.csv")) + list(EXT.glob("*.json")):
        shutil.copy2(f, OUT / "results/external" / f.name)
    for f in (ROOT / "paper/latex/tables").glob("*.tex"):
        shutil.copy2(f, OUT / "results/paper_tables" / f.name)
    for f in (ROOT / "outputs/reports/paper_tables").glob("*.csv"):
        shutil.copy2(f, OUT / "results/paper_tables" / f.name)
    for folder, names in FINAL_LOGS.items():
        for n in names:
            shutil.copy2(folder / n, OUT / "logs" / n)
    (OUT / "logs/README.md").write_text(
        "# Logs of the final runs\n\n"
        "- `run_full_pipeline.log` — stage timestamps of the full BigFlow run (2026-10-09/10, 66.9M flows).\n"
        "- `run_label_ambiguity.log`, `run_build_sample.log`, `run_part1.log`, `run_part5.log`, "
        "`run_robustA.log`, `run_robustB.log`, `run_egraphsage.log`, `run_figures.log` — the stages of that run.\n"
        "- `run_part34.log` — parts 3 and 4 of that run. **Its Part 3 numbers for the 10-minute "
        "time-block split are superseded**: the block split was fixed on 2026-10-10 and Part 3 re-run, "
        "see `run_part3_blockfix.log` (the result CSVs contain the re-run). Part 4 is unaffected.\n"
        "- `run_paper_tables.log` — table generation after the re-run.\n"
        "- `run_unsw.log`, `run_ciciot.log` — external validation (2026-10-11).\n"
        "- Logs of runs on an earlier 46-file subset of BigFlow are not included; their results are "
        "superseded.\n")


# ======================================================================
# Final paper-2 export (frozen protocol): python scripts/make_export.py --final
# ======================================================================

F2 = ROOT / "outputs/paper2_final"
OUT2 = F2 / "paper2_final_export"
MAIN_P = "episodes gap 1 min, embargo 5 min"


def _ci(v):
    from src.stats.protocol import bootstrap_ci
    v = np.asarray(v, float)
    lo, hi = bootstrap_ci(v)
    return f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def _fpr(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} mean / {v.max():.3f} worst seed"


def _test(tests, family, subset, **keys):
    t = tests[tests.family.str.startswith(family) & tests.subset.str.startswith(subset)
              & (tests.metric == "TPR@FPR1pct")]
    for k, v in keys.items():
        t = t[t[k] == v]
    r = t.iloc[0]
    p = "< 0.001" if r.p_holm < 0.001 else f"= {r.p_holm:.3f}"
    return f"{r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}], Holm p {p}"


def final_summary():
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from paper2_stats import load
    ctx, gat = load("context"), load("gated")
    tests = pd.read_csv(F2 / "stats/tests_all_families.csv")
    tab = lambda folder, name: pd.read_csv(F2 / "paper_tables" / folder / f"{name}.csv")  # noqa: E731
    meta = json.loads((F2 / "logs/context_bigflow_final_ctx_main_meta.json").read_text())
    hist = lambda src, rep, sub: ctx[(ctx.source == src) & (ctx.protocol == MAIN_P) & (ctx.model == "HistGB")  # noqa: E731
                                     & (ctx.representation == rep) & (ctx.subset == sub)]
    cic_all, cic_i = hist("CSE-CIC-IDS2018", "flow", "all"), hist("CSE-CIC-IDS2018", "flow", "cross-episode")
    ton_all, ton_i = hist("ToN-IoT", "flow", "all"), hist("ToN-IoT", "flow", "cross-episode")
    cic_ii = hist("CSE-CIC-IDS2018", "flow", "fallback")
    ton_host = hist("ToN-IoT", "flow + host", "all")
    ton_nh = hist("ToN-IoT", "flow + neighbourhood + host", "all")
    gated_ton = gat[(gat.source == "ToN-IoT") & (gat.validation == "split validation")
                    & (gat.representation == "family-gated") & (gat.subset == "all")]
    chrono = ctx[(ctx.source == "ToN-IoT") & (ctx.protocol == "chronological, embargo 5 min") & (ctx.model == "HistGB")
                 & (ctx.representation == "flow") & (ctx.subset == "all")]
    s1 = tab("exploratory", "S1_same_family")
    s4 = tab("exploratory", "S4_ambiguity_removed")
    t9b = tab("supplement", "A10_episode_uncertainty")

    lines = [
        "# Paper 2 — final results summary (frozen protocol)\n",
        f"Generated {date.today().isoformat()} by `scripts/make_export.py --final`. Every number below is computed "
        "from the CSV files shipped in this export (`results/`, `stats/`, `paper_tables/`).\n",
        "## Scope\n",
        "- Paper 2 covers PhD research objective 1 only: \"To design AI-based intelligent model to enhance attack "
        "detection accuracy and reduce false alarms in Security Information and Event Management System\".",
        "- **Response-time metrics** (per-episode time to detect, episode detection rate, benign alerts per hour, "
        "per-flow inference latency) were **pre-specified in `PROTOCOL_FREEZE.md` (section 6) before any result "
        "was seen, and are reported in a companion paper (paper 3)**. They are not reported in paper 2. Metrics "
        "(a)–(c) were computed from the saved per-flow predictions into `outputs/paper3_baseline/`; latency is "
        "measured in paper 3.",
        "- Per-flow validation and test predictions for every run and seed are kept permanently in "
        "`outputs/paper2_final/predictions/` (not shipped here, about 1 GB).",
        "- Main sources: ToN-IoT and CSE-CIC-IDS2018 from BigFlow-NIDS v2. BoT-IoT and BigFlow's UNSW-NB15 part are "
        "limitations only. External: original UNSW-NB15 (full protocol), CIC-IoT2023 (random split only).\n",
        "## Run\n",
        f"- Protocol frozen in `PROTOCOL_FREEZE.md`; corrections after the freeze are listed in its section 9. Test "
        "families declared in `reports/stats_protocol.md` before any statistic was computed.",
        f"- Run: option (b-ii), launched from commit e800706 on 2026-10-16, finished 2026-10-19; meta of the main "
        f"context run records commit {meta['git_commit'][:7]}, scikit-learn {meta['scikit-learn']}, torch "
        f"{meta['torch']}, Python {meta['python']}.",
        "- Completeness verified: `reports/phase_a_completeness.md`, `reports/phase_b_completeness.md`.",
        "- Metric: design-weighted TPR at 1% FPR (threshold = 99th percentile of validation benign scores), shown "
        "with the FPR reached on test. Subset (i) = families testable on unseen episodes (primary); (ii) = "
        "single-episode families tested within their episode; (iii) = all test attacks.\n",
        "## What survives (frozen protocol)\n",
        f"**(a) Pooled rates mostly reflect test composition** (Tables 2a/2b, Figures 2 and 6). HistGB on flow "
        f"features, 30 seeds: CSE-CIC-IDS2018 (iii) {_ci(cic_all['TPR@FPR1pct'])}, (i) {_ci(cic_i['TPR@FPR1pct'])}, "
        f"(ii) {_ci(cic_ii['TPR@FPR1pct'])}; FPR reached {_fpr(cic_all['testFPR@1pct'])}. ToN-IoT (iii) "
        f"{_ci(ton_all['TPR@FPR1pct'])}, (i) {_ci(ton_i['TPR@FPR1pct'])}; FPR {_fpr(ton_all['testFPR@1pct'])}. "
        "On CSE-CIC the overall rate is dominated by single-episode families tested within their own episode; "
        "Figure 6 shows its seeds falling into clusters by which Infiltration episodes reach test.\n",
        f"**(b) Thresholds do not hold on test** (Table 3, Figure 3). ToN-IoT, target 1%: flow "
        f"{_fpr(ton_all['testFPR@1pct'])}; flow + host {_fpr(ton_host['testFPR@1pct'])}; flow + neighbourhood + host "
        f"{_fpr(ton_nh['testFPR@1pct'])}; family-gated (split validation) {_fpr(gated_ton['testFPR@1pct'])}.\n",
        "**(c) Context-feature gains were partly leakage** (Tables 5, 6, Figure 4). Subset (i), HistGB, 30 seeds: "
        f"k-NN neighbourhood on CSE-CIC {_test(tests, 'F2', '(i)', source='CSE-CIC-IDS2018', model='HistGB', candidate='flow + neighbourhood')}; "
        f"host on CSE-CIC {_test(tests, 'F2', '(i)', source='CSE-CIC-IDS2018', model='HistGB', candidate='flow + host')}; "
        f"host on ToN-IoT {_test(tests, 'F2', '(i)', source='ToN-IoT', model='HistGB', candidate='flow + host')}.\n",
        f"**(d) ToN-IoT drift** (Table A7, Figure 5). Chronological split vs main protocol, (iii): HistGB flow "
        f"{_ci(chrono['TPR@FPR1pct'])}; difference {_test(tests, 'F4', '(iii)', source='ToN-IoT', candidate='chronological, embargo 5 min')}.\n",
        "**(e) Few episodes, wide uncertainty** (Table A10, Figure 6).\n",
        md(t9b),
        "\n## Exploratory analyses (post-hoc, labelled as deviations)\n",
        "Not pre-specified; reported as exploratory (Tables S1–S4).\n",
        "S1 — same families under the random split (no detectable penalty for unseen episodes at this number of "
        "episodes):\n", md(s1),
        "\nS4 — benign-identical flows removed from subset (i) (duplicates do not explain the subset (i) rates):\n",
        md(s4),
        "\nS2 (split-gap decomposition, both orders and average) and S3 (tied thresholds, score ≥ τ) are in "
        "`paper_tables/exploratory/`.\n",
        "## Reporting rules used throughout\n",
        "- Every TPR at 1% FPR is shown with the FPR reached on test; every pooled subset value has a per-family table.",
        "- Cells whose threshold equals the maximum score read \"cannot operate at 1% FPR (tied scores)\".",
        "- The family-gated detector is reported as a safe default; it does not match the best single feature set "
        "under split validation.\n",
        "## Contents\n",
        "- `PROTOCOL_FREEZE.md`, `reports/` (stats protocol, phase-1 summary, completeness checks, post-hoc reports, "
        "runtime tables, endpoint selection)",
        "- `paper_tables/` (main Tables 1–7, supplement A1–A13, exploratory S1–S4; CSV + LaTeX + README preview)",
        "- `figures/` (Figures 1–6, PDF + PNG)",
        "- `stats/` (all tests, descriptive tables, per-family TPR, post-hoc CSVs)",
        "- `results/` (per-seed metric rows of every final run), `logs/` (run logs and per-run meta with commit, "
        "versions and wall-clock time)",
        "- `paper/` (outline, related work), `scripts/` (driver, statistics, tables, figures, launch script)",
    ]
    return "\n".join(lines) + "\n"


def build_final_export():
    import fnmatch
    if OUT2.exists():
        shutil.rmtree(OUT2)  # generated folder, rebuilt from committed sources
    OUT2.mkdir(parents=True)
    copies = {
        ROOT / "PROTOCOL_FREEZE.md": "PROTOCOL_FREEZE.md",
        ROOT / "paper/outline.md": "paper/outline.md", ROOT / "paper/related_work.md": "paper/related_work.md",
    }
    for d, pat in ((F2 / "reports", "*"), (F2 / "stats", "*.csv"), (F2 / "figures", "*"),
                   (F2 / "results", "*_final_*.csv"), (F2 / "logs", "*final*"), (F2 / "logs", "tests_*.log")):
        for f in d.glob(pat):
            if f.is_file():
                copies[f] = f"{d.name}/{f.name}"
    for f in (F2 / "paper_tables").rglob("*"):
        if f.is_file():
            copies[f] = str(f.relative_to(F2))
    for name in ("paper2_final.py", "paper2_stats.py", "make_paper2_tables.py", "make_paper2_figures.py",
                 "run_paper2_final.sh", "select_endpoint.py", "per_family_tpr.py", "posthoc_split_decomposition.py",
                 "posthoc_ambiguity_removed.py", "tie_sensitivity.py", "check_phase_a.py", "check_phase_b.py",
                 "build_stratified_sample.py", "make_export.py"):
        copies[ROOT / "scripts" / name] = f"scripts/{name}"
    for src, rel in copies.items():
        dst = OUT2 / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (OUT2 / "results_summary.md").write_text(final_summary())
    zip_path = F2 / "paper2_final_export.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT2.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(F2))
    n = sum(1 for f in OUT2.rglob("*") if f.is_file())
    print(f"{OUT2} ({n} files) -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


def main():
    import sys
    if "--final" in sys.argv:
        build_final_export()
        return
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    copy_files()
    (OUT / "results_summary.md").write_text(build_summary())
    zip_path = ROOT / "paper2_export.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(ROOT))
    n = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"{OUT} ({n} files) -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

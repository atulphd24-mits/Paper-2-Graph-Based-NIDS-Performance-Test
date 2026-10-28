"""
Publication-ready LaTeX tables (booktabs) built from the leakage-study CSVs.

Output: paper/latex/tables/*.tex, included by paper/latex/main.tex.
Every number comes from outputs/reports/leakage_study/ - re-run this script after any
experiment changes.

Usage:
    python scripts/make_paper_tables.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R = PROJECT_ROOT / "outputs/reports/leakage_study"
OUT = PROJECT_ROOT / "paper/latex/tables"

SRC = {"ToN-IoT": "ToN-IoT", "CSE-CIC-IDS2018": "CSE-CIC-IDS2018",
       "BoT-IoT": "BoT-IoT", "UNSW-NB15": "UNSW-NB15"}
MODEL = {"HistGB": "Gradient boosting", "RandomForest": "Random forest", "MLP": "MLP"}
REP = {"flow": "Flow", "flow + neighbourhood": "Flow + neighbourhood",
       "flow + host": "Flow + host", "flow + neighbourhood + host": "Flow + neigh. + host"}


def esc(text):
    return (str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


def pm(mean, std, bold=False, digits=3):
    if pd.isna(mean):
        return "--"
    body = f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"
    return f"\\textbf{{{body}}}" if bold else body


def stars(p):
    if pd.isna(p):
        return ""
    return "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def delta(x, p=None):
    if pd.isna(x):
        return "--"
    return f"{x:+.3f}{stars(p) if p is not None else ''}"


def write(name, caption, label, colspec, header, rows, notes=None, size="\\small"):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        size,
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        is_rule = row.startswith(("\\midrule", "\\cmidrule", "\\addlinespace"))
        lines.append(row if is_rule or row.endswith("\\\\") else row + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\footnotesize {notes}}}")
    lines.append("\\end{table}")
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    check(name, text, colspec)
    (OUT / f"{name}.tex").write_text(text)


def check(name, text, colspec):
    """Every data row must have exactly one cell per column and end with \\\\."""
    body = text.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith(("\\midrule", "\\cmidrule", "\\addlinespace")):
            assert line.endswith("\\\\"), f"{name}: row without line ending: {line}"
    ncols = sum(colspec.count(c) for c in "lcrp")
    for line in text.splitlines():
        if line.endswith("\\\\") and "multicolumn" not in line:
            cells = line[:-2].count("&") + 1 - line.count("\\&")
            assert cells == ncols, f"{name}: {cells} cells, expected {ncols}: {line}"


# ----------------------------------------------------------------------
# Table 1 - composition and label ambiguity per source
# ----------------------------------------------------------------------

def table_sources():
    src = pd.read_csv(R / "label_ambiguity_by_source.csv")
    fam = pd.read_csv(R / "label_ambiguity_by_family.csv")
    families = fam.groupby("source").size()
    rows = []
    for r in src.sort_values("flows", ascending=False).itertuples():
        rows.append(" & ".join([
            SRC[r.source], f"{r.flows:,}", f"{r.attack_share * 100:.1f}",
            str(families[r.source]), f"{r.distinct_vectors:,}",
            f"{r.attack_flows_in_ambiguous_vectors * 100:.1f}", f"{r.irreducible_error * 100:.2f}",
        ]))
    total = src["flows"].sum()
    rows.append("\\midrule")
    rows.append(f"Total & {total:,} & {np.average(src.attack_share, weights=src.flows) * 100:.1f} "
                f"& {families.sum()} & {src.distinct_vectors.sum():,} & -- & --")
    write("tab_sources",
          "Composition of BigFlow-NIDS by recovered source dataset and exact-duplicate label "
          "ambiguity. A flow is \\emph{ambiguous} when its behaviour vector (timestamps, IP "
          "addresses, source port and DNS query id removed) also occurs with the other label. "
          "The irreducible error is the lowest error any per-flow classifier can reach on these "
          "features.",
          "tab:sources", "lrrrrrr",
          "Source & Flows & Attack (\\%) & Families & Distinct vectors & "
          "Ambig.\\ attacks (\\%) & Irreducible err.\\ (\\%)", rows)


# ----------------------------------------------------------------------
# Table 2 - ambiguity per attack family (main sources)
# ----------------------------------------------------------------------

def table_families():
    fam = pd.read_csv(R / "label_ambiguity_by_family.csv")
    rows = []
    for source in ["ToN-IoT", "CSE-CIC-IDS2018"]:
        d = fam[fam.source == source].sort_values("share_identical_to_benign", ascending=False)
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{SRC[source]}}}}} \\\\")
        for r in d.itertuples():
            rows.append(" & ".join([
                "\\quad " + esc(r.attack.replace("_-", " ").replace("_", " ")
                                .replace("Infilteration", "Infiltration")),
                f"{r.flows:,}", f"{r.distinct_vectors:,}",
                f"{r.share_identical_to_benign * 100:.1f}",
            ]))
    write("tab_families",
          "Share of attack flows whose exact behaviour vector also occurs as benign, per attack "
          "family (all flows). Families with a single distinct vector (e.g.\\ FTP-BruteForce) are "
          "fully duplicated.",
          "tab:families", "lrrr",
          "Attack family & Flows & Distinct vectors & Identical to benign (\\%)", rows,
          size="\\footnotesize")


# ----------------------------------------------------------------------
# Table 3 - shortcut features on pooled data
# ----------------------------------------------------------------------

def table_shortcuts():
    p1 = pd.read_csv(R / "part1_shortcuts_raw.csv")
    sid = p1[p1.task == "identify source from behaviour"]["balanced_accuracy"]
    det = p1[p1.task == "pooled attack detection"].groupby("Model")[["F1", "PR_AUC", "ROC_AUC"]]
    mean, std = det.mean(), det.std()
    single = pd.read_csv(R / "part1_single_feature_source_id.csv").head(3)
    order = [("source prior only", "Source dataset only (one-hot)"),
             ("timestamp only", "Flow start timestamp only"),
             ("behaviour features", "All behaviour features")]
    rows = []
    for key, label in order:
        rows.append(" & ".join([label] + [pm(mean.loc[key, m], std.loc[key, m])
                                          for m in ["F1", "PR_AUC", "ROC_AUC"]]))
    best = ", ".join(f"\\texttt{{{esc(r.feature)}}} ({r.source_balanced_accuracy * 100:.1f}\\%)"
                     for r in single.itertuples())
    write("tab_shortcuts",
          "Shortcut features on pooled BigFlow-NIDS (random row split, 3 seeds, mean $\\pm$ std). "
          f"Behaviour features alone identify the source dataset with "
          f"{sid.mean() * 100:.1f}\\,$\\pm$\\,{sid.std() * 100:.2f}\\% balanced accuracy.",
          "tab:shortcuts", "lccc",
          "Model input & F1 & PR-AUC & ROC-AUC", rows,
          notes=f"Best single features for source identification: {best}.")


# ----------------------------------------------------------------------
# Table 4 - split strategy (gradient boosting)
# ----------------------------------------------------------------------

def table_splits():
    s = pd.read_csv(R / "part3_splits_summary.csv")
    splits = [("random rows", "Random rows"), ("10-min time blocks", "10-min time blocks"),
              ("attack episodes", "Attack episodes")]
    rows = []
    for source in ["ToN-IoT", "CSE-CIC-IDS2018"]:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{SRC[source]}}}}} \\\\")
        for key, label in splits:
            b = s[(s.source == source) & (s.split == key) & (s.representation == "behaviour")].iloc[0]
            t = s[(s.source == source) & (s.split == key) & (s.representation == "timestamp only")].iloc[0]
            rows.append(" & ".join([
                "\\quad " + label,
                pm(b["F1_w_mean"], b["F1_w_std"]),
                pm(b["TPR@FPR1pct_mean"], b["TPR@FPR1pct_std"]),
                f"{b['dup_overlap_mean'] * 100:.1f}",
                pm(t["TPR@FPR1pct_mean"], t["TPR@FPR1pct_std"]),
            ]))
    write("tab_splits",
          "Effect of the split strategy (gradient boosting, 5 seeds). F1 is weighted to each "
          "source's real attack prevalence; TPR is the detection rate at 1\\% FPR with the alert "
          "threshold set on validation benign traffic. \\emph{Dup.} is the share of test flows "
          "whose exact behaviour vector occurs in training.",
          "tab:splits", "lcccc",
          "Split & F1 (weighted) & TPR@1\\%FPR & Dup.\\ (\\%) & TPR@1\\%FPR, timestamp only",
          rows)


# ----------------------------------------------------------------------
# Table 5 - split inflation across model families
# ----------------------------------------------------------------------

def table_split_models():
    hgb = pd.read_csv(R / "part3_splits_summary.csv")
    hgb = hgb[hgb.representation == "behaviour"].assign(model="HistGB")
    a1 = pd.read_csv(R / "robustA1_splits_models_summary.csv")
    eg = pd.read_csv(R / "egraphsage_summary.csv")
    eg = eg[eg.model == "E-GraphSAGE (ip_port)"].assign(model="E-GraphSAGE")
    d = pd.concat([hgb, a1, eg], ignore_index=True)
    names = dict(MODEL, **{"E-GraphSAGE": "E-GraphSAGE (IP:port)"})
    rows = []
    for source in ["ToN-IoT", "CSE-CIC-IDS2018"]:
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{SRC[source]}}}}} \\\\")
        for model in ["HistGB", "RandomForest", "MLP", "E-GraphSAGE"]:
            x = d[(d.source == source) & (d.model == model)].set_index("split")
            rnd, epi = x.loc["random rows"], x.loc["attack episodes"]
            rows.append(" & ".join([
                "\\quad " + names[model],
                pm(rnd["F1_w_mean"], rnd["F1_w_std"]),
                pm(epi["F1_w_mean"], epi["F1_w_std"]),
                f"{(rnd['F1_w_mean'] - epi['F1_w_mean']) * 100:+.1f}",
            ]))
    write("tab_split_models",
          "Weighted F1 under a random row split and an attack-episode split for four model "
          "families (gradient boosting, random forest and MLP: 5 seeds; E-GraphSAGE: 10 seeds). "
          "Inflation is the difference in F1 points.",
          "tab:split-models", "lccc",
          "Model & Random rows & Attack episodes & Inflation (pts)", rows)


# ----------------------------------------------------------------------
# Table 6 - context features across model families
# ----------------------------------------------------------------------

def table_context():
    hgb = pd.read_csv(R / "part4_graph_summary.csv").assign(model="HistGB")
    hgb_t = pd.read_csv(R / "part4_graph_significance.csv").assign(model="HistGB")
    oth = pd.read_csv(R / "robustA2_features_models_summary.csv")
    oth_t = pd.read_csv(R / "robustA2_features_models_significance.csv")
    s = pd.concat([hgb, oth], ignore_index=True)
    t = pd.concat([hgb_t, oth_t], ignore_index=True)
    t = t[t.metric == "TPR@FPR1pct"]
    reps = ["flow", "flow + neighbourhood", "flow + host", "flow + neighbourhood + host"]
    rows = []
    for source in ["CSE-CIC-IDS2018", "ToN-IoT"]:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{SRC[source]}}}}} \\\\")
        for model in ["HistGB", "RandomForest", "MLP"]:
            x = s[(s.source == source) & (s.model == model)].set_index("representation")
            best = x.loc[reps, "TPR@FPR1pct_mean"].max()
            cells = ["\\quad " + MODEL[model]]
            for rep in reps:
                m = x.loc[rep, "TPR@FPR1pct_mean"]
                p = t[(t.source == source) & (t.model == model) & (t.representation == rep)]["wilcoxon_p"]
                cell = pm(m, x.loc[rep, "TPR@FPR1pct_std"], bold=np.isclose(m, best))
                cells.append(cell + (stars(p.iloc[0]) if len(p) else ""))
            rows.append(" & ".join(cells))
    write("tab_context",
          "Detection rate at 1\\% FPR with flow features and with graph-neighbourhood and "
          "host-behaviour context (attack-episode split, 10 seeds, mean $\\pm$ std). Bold: best "
          "per row. $^{*}$\\,$p<0.05$, $^{**}$\\,$p<0.01$, paired Wilcoxon signed-rank test "
          "against flow features.",
          "tab:context", "lcccc",
          "Model & Flow & + Neighbourhood & + Host & + Neigh.\\ + host", rows,
          size="\\footnotesize")


# ----------------------------------------------------------------------
# Table 7 - E-GraphSAGE
# ----------------------------------------------------------------------

def table_egraphsage():
    s = pd.read_csv(R / "egraphsage_summary.csv")
    t = pd.read_csv(R / "egraphsage_significance.csv")
    t = t[t.metric == "TPR@FPR1pct"]
    models = [("HistGB (flow, no graph)", "Gradient boosting, flow only (no graph)"),
              ("E-GraphSAGE (ip)", "E-GraphSAGE, IP nodes"),
              ("E-GraphSAGE (ip_port)", "E-GraphSAGE, IP:port nodes"),
              ("E-GraphSAGE (random)", "E-GraphSAGE, random nodes (control)")]
    rows = []
    for source in ["CSE-CIC-IDS2018", "ToN-IoT"]:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{SRC[source]}}}}} \\\\")
        for key, label in models:
            cells = ["\\quad " + label]
            for split in ["random rows", "attack episodes"]:
                x = s[(s.source == source) & (s.split == split) & (s.model == key)].iloc[0]
                p = t[(t.source == source) & (t.split == split) & (t.model == key)]["wilcoxon_p"]
                cells.append(pm(x["TPR@FPR1pct_mean"], x["TPR@FPR1pct_std"])
                             + (stars(p.iloc[0]) if len(p) else ""))
                cells.append(f"{x['PR_AUC_w_mean']:.3f}")
            rows.append(" & ".join(cells))
    write("tab_egraphsage",
          "E-GraphSAGE (re-implemented) with three endpoint definitions against a flow-only "
          "gradient-boosting baseline (10 seeds). The random-node control gives every flow its "
          "own endpoints, removing shared graph structure. Significance markers: paired "
          "Wilcoxon test against the baseline.",
          "tab:egraphsage", "lcccc",
          "Model & \\multicolumn{2}{c}{Random rows} & \\multicolumn{2}{c}{Attack episodes} \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & TPR@1\\%FPR & PR-AUC & TPR@1\\%FPR & PR-AUC",
          rows, size="\\footnotesize")


# ----------------------------------------------------------------------
# Table 8 - family-gated detector
# ----------------------------------------------------------------------

def table_gated():
    s = pd.read_csv(R / "part5_two_branch_summary.csv")
    t = pd.read_csv(R / "part5_two_branch_significance.csv")
    t = t[t.metric == "TPR@FPR1pct"].set_index(["source", "baseline"])
    order = [("flow", "Flow only"), ("flow + neighbourhood", "Flow + neighbourhood"),
             ("flow + host", "Flow + host"), ("ensemble (mean of experts)", "Mean of the three experts"),
             ("family-gated (ours)", "Family-gated (ours)")]
    rows = []
    for source in ["CSE-CIC-IDS2018", "ToN-IoT"]:
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{SRC[source]}}}}} \\\\")
        x = s[s.source == source].set_index("representation")
        best = x.loc[[k for k, _ in order], "TPR@FPR1pct_mean"].max()
        for key, label in order:
            m = x.loc[key, "TPR@FPR1pct_mean"]
            rows.append(" & ".join([
                "\\quad " + label,
                pm(m, x.loc[key, "TPR@FPR1pct_std"], bold=np.isclose(m, best)),
                f"{x.loc[key, 'PR_AUC_w_mean']:.3f}",
                f"{x.loc[key, 'F1_w_mean']:.3f}",
            ]))
    note_parts = []
    for source in ["CSE-CIC-IDS2018", "ToN-IoT"]:
        for base, label in [("flow", "flow only"), ("flow + host", "flow + host"),
                            ("ensemble (mean of experts)", "expert mean")]:
            r = t.loc[(source, base)]
            note_parts.append(f"{SRC[source]} vs {label}: {r.mean_diff:+.3f} "
                              f"({int(r.wins)}/10 seeds, $p={r.wilcoxon_p:.3f}$)")
    write("tab_gated",
          "Family-gated detector against single feature sets and an unweighted expert ensemble "
          "(attack-episode split, 10 seeds). Bold: best per source.",
          "tab:gated", "lccc",
          "Detector & TPR@1\\%FPR & PR-AUC & F1 (weighted)", rows,
          notes="Paired differences in TPR@1\\%FPR for the gated detector: "
                + "; ".join(note_parts) + ".")


# ----------------------------------------------------------------------
# Table 9 - episode gap sensitivity
# ----------------------------------------------------------------------

def table_gap():
    s = pd.read_csv(R / "robustB_gap_summary.csv")
    t = pd.read_csv(R / "robustB_gap_significance.csv")
    t = t[t.metric == "TPR@FPR1pct"]
    reps = ["flow", "flow + host", "flow + neighbourhood + host"]
    rows = []
    for source in ["CSE-CIC-IDS2018", "ToN-IoT"]:
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{SRC[source]}}}}} \\\\")
        for gap in [1, 5, 30]:
            x = s[(s.source == source) & (s.gap_min == gap)].set_index("representation")
            cells = [f"\\quad {gap} min"]
            for rep in reps:
                p = t[(t.source == source) & (t.gap_min == gap) & (t.representation == rep)]["wilcoxon_p"]
                cells.append(pm(x.loc[rep, "TPR@FPR1pct_mean"], x.loc[rep, "TPR@FPR1pct_std"])
                             + (stars(p.iloc[0]) if len(p) else ""))
            rows.append(" & ".join(cells))
    write("tab_gap",
          "Sensitivity to the episode gap (gradient boosting, 10 seeds, TPR at 1\\% FPR, "
          "mean $\\pm$ std). Markers: paired Wilcoxon test against flow features at the same gap.",
          "tab:gap", "lccc",
          "Episode gap & Flow & + Host & + Neigh.\\ + host", rows, size="\\footnotesize")


# ----------------------------------------------------------------------
# Table 10 - cross-dataset summary (BigFlow sources + external datasets)
# ----------------------------------------------------------------------

EXT = PROJECT_ROOT / "outputs/reports/external"


def _context_delta(sig, model, rep):
    row = sig[(sig.metric == "TPR@FPR1pct") & (sig.model == model) & (sig.representation == rep)]
    if row.empty:
        return "n/a"
    r = row.iloc[0]
    return f"{r.mean_diff:+.3f}{stars(r.wilcoxon_p)}"


def table_external():
    rows = []
    p1 = pd.read_csv(R / "part1_shortcuts_raw.csv")
    source_id = p1[p1.task == "identify source from behaviour"]["balanced_accuracy"].mean()
    amb = pd.read_csv(R / "label_ambiguity_by_source.csv").set_index("source")
    p3 = pd.read_csv(R / "part3_splits_summary.csv")
    p4 = pd.read_csv(R / "part4_graph_significance.csv").assign(model="HistGB")
    eg = pd.read_csv(R / "egraphsage_significance.csv")

    def split_cells(summary, key):
        b = summary[summary.representation == "behaviour"].set_index("split")
        t = summary[summary.representation == "timestamp only"].set_index("split")
        f1 = f"{b.loc['random rows', 'F1_w_mean']:.3f} / {b.loc['attack episodes', 'F1_w_mean']:.3f}"
        ts = f"{t.loc['random rows', 'TPR@FPR1pct_mean']:.2f} / {t.loc['attack episodes', 'TPR@FPR1pct_mean']:.2f}"
        return ts, f1

    for source in ["ToN-IoT", "CSE-CIC-IDS2018"]:
        ts, f1 = split_cells(p3[p3.source == source], source)
        sig = p4[p4.source == source]
        e = eg[(eg.source == source) & (eg.split == "attack episodes") & (eg.metric == "TPR@FPR1pct")
               & (eg.model == "E-GraphSAGE (ip)")].iloc[0]
        rows.append(" & ".join([
            f"BigFlow: {SRC[source]}", f"source dataset ({source_id * 100:.1f})",
            f"{amb.loc[source, 'attack_flows_in_ambiguous_vectors'] * 100:.1f}", ts, f1,
            _context_delta(sig, "HistGB", "flow + neighbourhood"),
            _context_delta(sig, "HistGB", "flow + host"),
            f"{e.mean_diff:+.3f}{stars(e.wilcoxon_p)}",
        ]))

    for tag, label in [("unsw", "UNSW-NB15 (original)"), ("ciciot", "CIC-IoT2023")]:
        amb_e = pd.read_csv(EXT / f"{tag}_ambiguity_summary.csv").iloc[0]
        single = pd.read_csv(EXT / f"{tag}_single_feature.csv").iloc[0]
        splits = pd.read_csv(EXT / f"{tag}_splits_summary.csv")
        sig = pd.read_csv(EXT / f"{tag}_context_significance.csv")
        if (splits.split == "attack episodes").any():
            ts, f1 = split_cells(splits, tag)
        else:
            b = splits[splits.representation == "behaviour"].set_index("split")
            ts, f1 = "n/a", f"{b.loc['random rows', 'F1_w_mean']:.3f} / n/a"
        ambiguity = (f"{amb_e.attack_flows_in_ambiguous_vectors * 100:.1f}"
                     if amb_e.duplicate_rows > 0.001 else "n/a$^{\\dagger}$")
        egr = sig[(sig.metric == "TPR@FPR1pct") & (sig.model == "E-GraphSAGE (ip)")]
        rows.append(" & ".join([
            label,
            f"\\texttt{{{esc(single.feature)}}} ({single.binary_balanced_accuracy * 100:.1f})",
            ambiguity, ts, f1,
            _context_delta(sig, "HistGB", "flow + neighbourhood"),
            _context_delta(sig, "HistGB", "flow + host"),
            (f"{egr.iloc[0].mean_diff:+.3f}{stars(egr.iloc[0].wilcoxon_p)}" if len(egr) else "n/a"),
        ]))

    write("tab_external",
          "Cross-dataset summary. \\emph{Shortcut}: strongest single input and its balanced "
          "accuracy in \\% (BigFlow: source identification on pooled data; external datasets: "
          "attack vs benign from one feature). \\emph{Timestamp-only}: TPR at 1\\% FPR under random "
          "rows / attack episodes. \\emph{F1}: weighted F1 of gradient boosting under random rows / "
          "attack episodes. Context and E-GraphSAGE columns: change in TPR at 1\\% FPR against "
          "flow-only gradient boosting (10 seeds; $^{*}p<0.05$, $^{**}p<0.01$). n/a: not applicable "
          "because the release has no timestamps or addresses.",
          "tab:external", "lccccccc",
          "Dataset & Shortcut (\\%) & Ambig.\\ (\\%) & Timestamp-only & F1 & + Neigh. & + Host & "
          "E-GraphSAGE (IP)",
          rows, size="\\scriptsize",
          notes="$^{\\dagger}$CIC-IoT2023 features are continuous window statistics: only 34 of "
                "46.7 million rows are exact duplicates, so exact-match ambiguity is not informative.")


FIGURES = {  # paper figure name -> file produced by scripts/make_paper_figures.py
    "fig_workflow.pdf": "fig6_workflow.pdf",
    "fig_ambiguity.pdf": "fig1_label_ambiguity.pdf",
    "fig_splits.pdf": "fig2_split_leakage.pdf",
    "fig_context_models.pdf": "fig5_context_by_model.pdf",
    "fig_family_delta.pdf": "fig4_family_delta.pdf",
}


def copy_figures():
    import shutil
    source = PROJECT_ROOT / "outputs/figures/paper"
    target = PROJECT_ROOT / "paper/latex/figures"
    target.mkdir(parents=True, exist_ok=True)
    for dst, src in FIGURES.items():
        shutil.copyfile(source / src, target / dst)


if __name__ == "__main__":
    copy_figures()
    table_sources()
    table_families()
    table_shortcuts()
    table_splits()
    table_split_models()
    table_context()
    table_egraphsage()
    table_gated()
    table_gap()
    table_external()
    print("Tables:", sorted(p.name for p in OUT.glob("*.tex")))

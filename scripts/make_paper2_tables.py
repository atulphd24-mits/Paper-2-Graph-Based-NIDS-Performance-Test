"""
Paper 2 tables (CSV + LaTeX + a Markdown preview) from the committed final-run files.

Rules (PROTOCOL_FREEZE.md and the writing decisions of 2026-10-20):
- frozen-protocol results are primary (tables T1-T11); post-hoc checks are in clearly labelled
  exploratory tables (S1-S3);
- every TPR at 1% FPR is shown with the FPR reached on test (mean, sd, worst seed);
- every pooled subset value has a companion per-family table (design-weighted per-family TPR and the
  family's share of the subset's test-attack weight), "b" tables;
- cells whose threshold is tied with the maximum score are shown as "cannot operate at 1% FPR (tied
  scores)", with the number of tied seeds;
- response-time metrics are not reported (paper 3).

Usage: python scripts/make_paper2_tables.py  -> outputs/paper2_final/paper_tables/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from paper2_stats import load  # noqa: E402
from src.models.predictions import slug  # noqa: E402
from src.stats.protocol import bootstrap_ci  # noqa: E402

F = ROOT / "outputs/paper2_final"
OUT = F / "paper_tables"
STATS = F / "stats"
V1 = F / "v1_baseline/leakage_study"
MAIN = "episodes gap 1 min, embargo 5 min"
BF = ["ToN-IoT", "CSE-CIC-IDS2018"]
TIED = "cannot operate at 1% FPR (tied scores)"
SUB = {"cross-episode": "(i)", "fallback": "(ii)", "all": "(iii)"}
PROTO_SHORT = {MAIN: "episodes (main)", "episodes gap 5 min, embargo 5 min": "episodes, 5-min gap",
               "episodes gap 1 min, embargo 0 min": "episodes, no embargo",
               "chronological, embargo 5 min": "chronological", "random rows": "random rows"}
TABLES = []
# build key -> (folder, published name). Main text: Tables 1-7; supplement: A1-A13; exploratory: S1-S4.
NAMES = {
    "T1_sources": ("main", "Table1_sources"), "T2_composition": ("main", "Table2a_composition"),
    "T2b_composition_by_family": ("main", "Table2b_composition_by_family"),
    "T3_threshold_transfer": ("main", "Table3_threshold_transfer"), "T4_split_comparison": ("main", "Table4_split_comparison"),
    "T5_context_features": ("main", "Table5_context_features"), "M6_model_comparison": ("main", "Table6_model_comparison"),
    "T11_old_vs_new": ("main", "Table7_old_vs_new"),
    "T4b_split_comparison_by_family": ("supplement", "A1_split_comparison_by_family"),
    "T5b_context_by_family": ("supplement", "A2_context_by_family"), "T6_egraphsage": ("supplement", "A3_egraphsage_all"),
    "T6b_egraphsage_by_family": ("supplement", "A4_egraphsage_by_family"), "T7_gated_detector": ("supplement", "A5_gated_all"),
    "T7b_gated_by_family": ("supplement", "A6_gated_by_family"), "T8_protocol_sensitivity": ("supplement", "A7_protocol_sensitivity"),
    "T8b_drift_by_family": ("supplement", "A8_drift_by_family"), "T9a_within_episode_families": ("supplement", "A9_within_episode_families"),
    "T9b_episode_uncertainty": ("supplement", "A10_episode_uncertainty"), "T10_external": ("supplement", "A11_external"),
    "T10b_external_by_family": ("supplement", "A12_external_by_family"), "T12_statistics_protocol": ("supplement", "A13_statistics_protocol"),
    "S1_same_family": ("exploratory", "S1_same_family"), "S2_split_gap_decomposition": ("exploratory", "S2_split_gap_decomposition"),
    "S3_tie_sensitivity": ("exploratory", "S3_tie_sensitivity"), "S4_ambiguity_removed": ("exploratory", "S4_ambiguity_removed"),
}


# ----------------------------------------------------------------------------------------- helpers

def fmt(v, d=3):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def ci(v):
    v = np.asarray(v, float)
    lo, hi = bootstrap_ci(v)
    return f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]" if len(v) > 1 else fmt(v.mean())


def fpr(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f} / {v.max():.3f}"


def pval(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def diff(row):
    return f"{row.mean_diff:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]"


TIES = pd.read_csv(STATS / "threshold_ties.csv")
TIE_COUNT = TIES.groupby(["part", "source", "protocol", "score"])["tie_at_max"].sum().to_dict()


def tied(part, source, protocol, score):
    return int(TIE_COUNT.get((part, slug(source), slug(protocol), score), 0))


def tpr_cell(values, n_tied):
    if n_tied:
        return f"{TIED}, {n_tied}/{len(values)} seeds"
    return ci(values)


def save(key, title, frame, note="", exploratory=False):
    folder, name = NAMES[key]
    (OUT / folder).mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / folder / f"{name}.csv", index=False)
    esc = lambda s: (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")  # noqa: E731
                     .replace("_", "\\_").replace("#", "\\#").replace("±", "$\\pm$").replace("≥", "$\\geq$")
                     .replace("−", "$-$").replace("<", "$<$").replace(">", "$>$"))
    cols = "l" * len(frame.columns)
    tex = ["\\begin{table*}[t]", "\\centering\\scriptsize",
           f"\\caption{{{esc(title)}{(' ' + esc(note)) if note else ''}}}", f"\\label{{tab:{name}}}",
           "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
           " & ".join(esc(c) for c in frame.columns) + " \\\\", "\\midrule"]
    tex += [" & ".join(esc(v) for v in r) + " \\\\" for r in frame.itertuples(index=False)]
    tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    (OUT / folder / f"{name}.tex").write_text("\n".join(tex) + "\n")
    TABLES.append((folder, name, title, note, frame, exploratory))


def per_family(part, sources, protocol, scores, subsets=("(i) cross-episode", "(iii) all"), seeds=None,
               validation="-", label_map=None):
    """Design-weighted per-family TPR (mean over seeds) and weight share, wide by score."""
    parts = [part] if isinstance(part, str) else list(part)
    pf = PF[PF.part.isin(parts) & PF.source.isin([slug(s) for s in sources]) & (PF.protocol == slug(protocol))
            & PF.score.isin(scores) & PF.subset.isin(subsets) & (PF.validation == validation)]
    if seeds is not None:
        pf = pf[pf.seed.isin(seeds)]
    g = pf.groupby(["source", "subset", "family", "score"]).agg(
        tpr=("tpr", "mean"), share=("weight_share", "mean"), seeds=("seed", "nunique"), tied=("tied", "sum")).reset_index()
    rows = []
    for (src, sub, fam), h in g.groupby(["source", "subset", "family"]):
        row = {"source": src.replace("_", "-").replace("CSE-CIC-IDS2018", "CSE-CIC-IDS2018"), "subset": sub,
               "family": fam, "seeds in subset": int(h.seeds.max()), "weight share": fmt(h.share.mean())}
        for sc in scores:
            r = h[h.score == sc]
            name = (label_map or {}).get(sc, sc)
            row[name] = "" if r.empty else (TIED if r.tied.iloc[0] else fmt(r.tpr.iloc[0]))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- data

SPL, CTX, GAT = load("splits"), load("context"), load("gated")
EGS = load("egraphsage")
TESTS = pd.read_csv(STATS / "tests_all_families.csv")
ORDERS = pd.read_csv(STATS / "posthoc_split_decomposition_orders.csv")
PF = pd.read_csv(STATS / "per_family_tpr.csv")
AMB_F = pd.read_csv(V1 / "label_ambiguity_by_family.csv")
AMB_S = pd.read_csv(V1 / "label_ambiguity_by_source.csv")
BOUNDS = pd.read_parquet(ROOT / "data/processed/episode_bounds_bigflow.parquet")
BOUNDS = BOUNDS[BOUNDS.gap_min == 1]


def test_row(family, subset, **keys):
    t = TESTS[(TESTS.family.str.startswith(family)) & (TESTS.subset.str.startswith(subset))
              & (TESTS.metric == "TPR@FPR1pct")]
    for k, v in keys.items():
        t = t[t[k] == v]
    return t.iloc[0] if len(t) == 1 else None


# ----------------------------------------------------------------------------------------- tables

def t1_data():
    rows = []
    for r in AMB_S.itertuples():
        b = BOUNDS[BOUNDS.source == r.source].groupby("attack").size()
        rows.append({"source": r.source, "role": "main" if r.source in BF else "limitation only",
                     "flows": f"{r.flows:,}", "attack share": fmt(r.attack_share), "attack families": len(b),
                     "families with ≥3 episodes (1-min gap)": int((b >= 3).sum()), "episodes": int(b.sum()),
                     "attack flows identical to benign": fmt(r.attack_flows_in_ambiguous_vectors),
                     "irreducible error": fmt(r.irreducible_error)})
    save("T1_sources", "BigFlow-NIDS sources: size, attack families, full-data episodes (1-min gap) and label "
         "ambiguity (attack flows whose exact feature vector also occurs as benign).", pd.DataFrame(rows),
         "Ambiguity from the v1 data audit (data property, unaffected by the fixes).")


def t2_composition():
    rows, fam_rows = [], []
    for src in BF:
        c = CTX[(CTX.source == src) & (CTX.protocol == MAIN) & (CTX.model == "HistGB") & (CTX.representation == "flow")]
        pf = PF[(PF.part.isin(["context_final_ctx_main", "context_final_hl_ctx_main"])) & (PF.source == slug(src))
                & (PF.protocol == slug(MAIN)) & (PF.score == "HistGB|flow")]
        for sub in ("cross-episode", "fallback", "all"):
            g = c[c.subset == sub]
            share = pf[pf.subset == f"{SUB[sub]} " + {"cross-episode": "cross-episode", "fallback": "fallback",
                                                         "all": "all"}[sub]]
            w_share = (share.groupby("seed").weight_share.sum()).mean() if len(share) else np.nan
            rows.append({"source": src, "subset": SUB[sub] + " " + sub, "seeds": g.seed.nunique(),
                         "families (mean)": fmt(g["share_of_test_attacks"].count() and
                                                share.groupby("seed").family.nunique().mean(), 1),
                         "test episodes (mean)": fmt(g.test_episodes.mean(), 1),
                         "share of test-attack flows": fmt(g.share_of_test_attacks.mean()),
                         "TPR@1%FPR [95% CI]": ci(g["TPR@FPR1pct"]), "FPR reached (mean ± sd / max)": fpr(g["testFPR@1pct"])})
    t = pd.DataFrame(rows)
    save("T2_composition", "Test composition (finding a): HistGB on flow features, main protocol, 30 seeds. "
         "Subset (i): families testable on unseen episodes; (ii): single-episode families tested within their "
         "own episode; (iii): all test attacks.", t,
         "Share of test-attack flows counts sampled flows; the per-family table gives design-weight shares.")
    fam = per_family(["context_final_ctx_main", "context_final_hl_ctx_main"], BF, MAIN, ["HistGB|flow"],
                     subsets=("(i) cross-episode", "(ii) fallback"), label_map={"HistGB|flow": "TPR@1%FPR"})
    amb = AMB_F.assign(source=AMB_F.source.map(slug))
    fam["identical to benign"] = [fmt(amb[(amb.source == slug(r.source)) & (amb.attack == r.family)]
                                      .share_identical_to_benign.squeeze()) for r in fam.itertuples()]
    fam["share not identical to benign"] = [fmt(1 - float(v)) if v != "" else "" for v in fam["identical to benign"]]
    fam["full-data episodes"] = [int(((BOUNDS.source == r.source) & (BOUNDS.attack == r.family)).sum()) for r in fam.itertuples()]
    save("T2b_composition_by_family", "Per-family TPR behind Table 2a (HistGB, flow features, main protocol, 30 "
         "seeds): design-weighted TPR, the family's share of the subset's test-attack weight, the share of the "
         "family's flows identical to benign flows, and its full-data episodes.", fam,
         "TPR and weight share are means over the seeds in which the family is in the subset, so shares can sum "
         "to slightly more than 1. CSE-CIC subset (i) is 94% Infiltration by weight (74.5% of Infiltration flows "
         "identical to benign flows); ToN-IoT scanning is 93.9% identical to benign. Removing these flows changes "
         "subset (i) TPR little on CSE-CIC (exploratory Table S4).")


def t3_thresholds():
    rows = []
    for src in BF:
        for proto in [MAIN, "episodes gap 1 min, embargo 0 min", "episodes gap 5 min, embargo 5 min",
                      "chronological, embargo 5 min"]:
            for rep in ["flow", "flow + neighbourhood", "flow + host", "flow + neighbourhood + host"]:
                g = CTX[(CTX.source == src) & (CTX.protocol == proto) & (CTX.model == "HistGB")
                        & (CTX.representation == rep) & (CTX.subset == "all")]
                if g.empty:
                    continue
                rows.append({"source": src, "protocol": PROTO_SHORT[proto], "model": "HistGB", "features": rep,
                             "seeds": g.seed.nunique(), "FPR reached, target 1%": fpr(g["testFPR@1pct"]),
                             "FPR reached, target 0.1%": fpr(g["testFPR@0.1pct"])})
        for model in ["RandomForest", "MLP"]:
            for rep in ["flow", "flow + host"]:
                g = CTX[(CTX.source == src) & (CTX.protocol == MAIN) & (CTX.model == model)
                        & (CTX.representation == rep) & (CTX.subset == "all")]
                rows.append({"source": src, "protocol": PROTO_SHORT[MAIN], "model": model, "features": rep,
                             "seeds": g.seed.nunique(), "FPR reached, target 1%": fpr(g["testFPR@1pct"]),
                             "FPR reached, target 0.1%": fpr(g["testFPR@0.1pct"])})
        g = GAT[(GAT.source == src) & (GAT.representation == "family-gated") & (GAT.subset == "all")
                & (GAT.validation == "split validation")]
        rows.append({"source": src, "protocol": PROTO_SHORT[MAIN], "model": "family-gated (split validation)",
                     "features": "experts", "seeds": g.seed.nunique(), "FPR reached, target 1%": fpr(g["testFPR@1pct"]),
                     "FPR reached, target 0.1%": fpr(g["testFPR@0.1pct"])})
    save("T3_threshold_transfer", "Thresholds do not hold on test (finding b): FPR reached on test when the threshold "
         "is set for 1% and 0.1% FPR on validation benign flows (mean ± sd / worst seed).", pd.DataFrame(rows),
         "Host features worsen transfer on ToN-IoT for HistGB only, not for RF or MLP (whose host cells are partly "
         "tied); the gated detector inherits the host expert's FPR.")


def t4_splits():
    rows = []
    for _, r in SPL[SPL.dataset == "bigflow"].groupby(["dataset", "source", "model", "representation", "protocol"]).size().reset_index().iterrows():
        g = SPL[(SPL.source == r.source) & (SPL.model == r.model) & (SPL.representation == r.representation)
                & (SPL.protocol == r.protocol) & (SPL.subset == "all")]
        part = {"bigflow": "splits_final_splits", "unsw": "splits_final_unsw_splits",
                "ciciot": "splits_final_ciciot_splits"}[r.dataset]
        n_tied = tied(part, r.source, r.protocol, f"{r.model}|{r.representation}")
        t = test_row("F1", "(iii)", source=r.source, model=r.model, representation=r.representation,
                     candidate=r.protocol) if r.protocol != MAIN else None
        dec = "–"
        if r.protocol == "random rows" and r.representation == "behaviour":
            o = ORDERS[(ORDERS.source == r.source) & (ORDERS.model == r.model)].set_index(["metric", "term"])["mean"]
            dec = (f"TPR {o[('TPR', 'mix (average)')]:+.3f} / {o[('TPR', 'threshold (average)')]:+.3f} / "
                   f"{o[('TPR', 'remainder')]:+.3f}; F1 {100 * o[('F1', 'mix (average)')]:+.1f} / "
                   f"{100 * o[('F1', 'threshold (average)')]:+.1f} / {100 * o[('F1', 'remainder')]:+.1f} pts")
        rows.append({"source": r.source, "model": r.model, "features": r.representation,
                     "split": PROTO_SHORT[r.protocol], "seeds": g.seed.nunique(),
                     "TPR@1%FPR (iii) [95% CI]": tpr_cell(g["TPR@FPR1pct"], n_tied),
                     "FPR reached": fpr(g["testFPR@1pct"]), "weighted F1": ci(g["F1_w"]),
                     "Δ vs episodes (main)": "– (tied)" if n_tied else (diff(t) if t is not None else "–"),
                     "p (Holm)": "– (tied)" if n_tied else (pval(t.p_holm) if t is not None else "–"),
                     "gap = family mix / threshold transfer / remainder (post-hoc, avg of both orders)": dec})
    order = {"random rows": 0, "episodes (main)": 1, "chronological": 2}
    d = pd.DataFrame(rows).sort_values(["source", "model", "features", "split"], key=lambda s: s.map(order) if s.name == "split" else s)
    save("T4_split_comparison", "Random rows and the chronological split vs the main episode split, all test attacks "
         "(iii), 10 seeds (frozen analysis; family F1). Timestamp-only is a control.", d,
         "The last column is a post-hoc decomposition (exploratory, Table S2), average of both correction orders.")
    fam = per_family("splits_final_splits", BF, MAIN, ["HistGB|behaviour"], subsets=("(iii) all",),
                     label_map={"HistGB|behaviour": "episodes (main)"})
    rnd = per_family("splits_final_splits", BF, "random rows", ["HistGB|behaviour"], subsets=("(iii) all",),
                     label_map={"HistGB|behaviour": "random rows"})
    chron = per_family("splits_final_splits", BF, "chronological, embargo 5 min", ["HistGB|behaviour"],
                       subsets=("(iii) all",), label_map={"HistGB|behaviour": "chronological"})
    m = fam.merge(rnd[["source", "family", "random rows", "weight share"]], on=["source", "family"], how="outer",
                  suffixes=("", " (random)")).merge(chron[["source", "family", "chronological"]], on=["source", "family"], how="outer")
    m = m.rename(columns={"weight share": "weight share (episodes)"}).drop(columns=["subset", "seeds in subset"])
    save("T4b_split_comparison_by_family", "Per-family TPR at 1% FPR behind Table 4 (HistGB, flow features, all test "
         "attacks, seeds 0-9), with each family's share of the test-attack weight under each split.", m)


def t5_context():
    rows = []
    for (ds, src, model), _ in CTX[CTX.protocol.isin([MAIN, "random rows"])].groupby(["dataset", "source", "model"]):
        proto = MAIN if ds != "ciciot" else "random rows"
        for rep in ["flow", "flow + neighbourhood", "flow + host", "flow + neighbourhood + host"]:
            base = CTX[(CTX.source == src) & (CTX.protocol == proto) & (CTX.model == model) & (CTX.representation == rep)]
            if base.empty:
                continue
            part = {"bigflow": "context_final_ctx_main", "unsw": "context_final_unsw_context",
                    "ciciot": "context_final_ciciot_context"}[ds]
            n_tied = tied(part, src, proto, f"{model}|{rep}")
            row = {"source": src, "model": model, "features": rep, "seeds": base.seed.nunique()}
            for sub in ("cross-episode", "all"):
                g = base[base.subset == sub]
                if g.empty:
                    row[f"TPR {SUB[sub]}"], row[f"Δ vs flow {SUB[sub]}"], row[f"p {SUB[sub]}"] = "–", "–", "–"
                    continue
                row[f"TPR {SUB[sub]}"] = tpr_cell(g["TPR@FPR1pct"], n_tied)
                t = test_row("F2", SUB[sub], source=src, model=model, candidate=rep) if rep != "flow" else None
                row[f"Δ vs flow {SUB[sub]}"] = "– (tied)" if n_tied else (diff(t) if t is not None else "–")
                row[f"p {SUB[sub]}"] = "– (tied)" if n_tied else (pval(t.p_holm) if t is not None else "–")
            row["FPR reached"] = fpr(base[base.subset == "all"]["testFPR@1pct"])
            rows.append(row)
    save("T5_context_features", "Context features vs flow features (finding c), main protocol (CIC-IoT2023: random "
         "split only). HistGB 30 seeds, RF and MLP 10 seeds; family F2, Holm within (source, model, subset).",
         pd.DataFrame(rows))
    reps = ["flow", "flow + neighbourhood", "flow + host", "flow + neighbourhood + host"]
    fam = per_family(["context_final_ctx_main", "context_final_hl_ctx_main"], BF, MAIN, [f"HistGB|{r}" for r in reps],
                     label_map={f"HistGB|{r}": r for r in reps})
    save("T5b_context_by_family", "Per-family TPR at 1% FPR behind Table 5 (HistGB, main protocol, 30 seeds).", fam,
         "Means over the seeds in which the family is in the subset.")


def t6_egraphsage():
    rows = []
    for (ds, src, proto, mode, model), g in EGS.groupby(["dataset", "source", "protocol", "mode", "model"]):
        g = g[g.seed < 10]
        row = {"source": src, "split": PROTO_SHORT[proto], "mode": mode, "model": model, "seeds": g.seed.nunique()}
        for sub in ("cross-episode", "all"):
            h = g[g.subset == sub]
            row[f"TPR {SUB[sub]}"] = ci(h["TPR@FPR1pct"]) if len(h) else "–"
            t = test_row("F6", SUB[sub], source=src, protocol=proto, mode=mode, candidate=model)
            row[f"Δ vs HistGB {SUB[sub]}"] = diff(t) if t is not None else "–"
            row[f"p {SUB[sub]}"] = pval(t.p_holm) if t is not None else "–"
            t7 = test_row("F7", SUB[sub], source=src, protocol=proto, model=model) if mode == "inductive" else None
            row[f"inductive − transductive {SUB[sub]}"] = diff(t7) if t7 is not None else "–"
        row["FPR reached"] = fpr(g[g.subset == "all"]["testFPR@1pct"])
        rows.append(row)
    sel = json.loads((F / "reports/endpoint_selection.json").read_text())["best_endpoint"]
    for src in BF:
        model = f"E-GraphSAGE ({sel[slug(src)]})"
        for m in (model, "HistGB (flow, no graph)"):
            g = EGS[(EGS.source == src) & (EGS.protocol == MAIN) & (EGS.model == m)
                    & (EGS["mode"].isin(["inductive", "-"]))]
            row = {"source": src, "split": "episodes (main), 30 seeds", "mode": "inductive" if m == model else "-",
                   "model": m + (" — endpoint chosen on validation" if m == model else ""), "seeds": g.seed.nunique()}
            for sub in ("cross-episode", "all"):
                h = g[g.subset == sub]
                row[f"TPR {SUB[sub]}"] = ci(h["TPR@FPR1pct"])
                t = test_row("F8", SUB[sub], source=src, candidate=model) if m == model else None
                row[f"Δ vs HistGB {SUB[sub]}"] = diff(t) if t is not None else "–"
                row[f"p {SUB[sub]}"] = pval(t.p_holm) if t is not None else "–"
                row[f"inductive − transductive {SUB[sub]}"] = "–"
            row["FPR reached"] = fpr(g[g.subset == "all"]["testFPR@1pct"])
            rows.append(row)
    save("T6_egraphsage", "E-GraphSAGE (IP, IP:port, random-endpoint control) vs flow-only HistGB on the same split, "
         "inductive and transductive (families F6, F7), and the 30-seed headline comparison (F8).", pd.DataFrame(rows))
    labels = {"HistGB (flow, no graph)": "HistGB (flow)", "E-GraphSAGE (ip)|inductive": "E-GraphSAGE IP",
              "E-GraphSAGE (ip_port)|inductive": "E-GraphSAGE IP:port", "E-GraphSAGE (random)|inductive": "random control"}
    fam = per_family("egraphsage_final_egs_ind", BF, MAIN, list(labels), label_map=labels)
    save("T6b_egraphsage_by_family", "Per-family TPR at 1% FPR behind Table A3 (inductive, main protocol, seeds 0-9).", fam)


def m6_models():
    """Main-text model comparison: E-GraphSAGE and the family-gated detector, main protocol only."""
    sel = json.loads((F / "reports/endpoint_selection.json").read_text())["best_endpoint"]
    rows = []

    def add(src, group, model, setting, g, comparison, t_i, t_iii):
        row = {"source": src, "group": group, "model or detector": model, "setting": setting, "seeds": g.seed.nunique(),
               "comparison": comparison}
        for sub, t in (("cross-episode", t_i), ("all", t_iii)):
            h = g[g.subset == sub]
            row[f"TPR {SUB[sub]}"] = ci(h["TPR@FPR1pct"])
            row[f"Δ {SUB[sub]}"] = diff(t) if t is not None else "–"
            row[f"p {SUB[sub]}"] = pval(t.p_holm) if t is not None else "–"
        row["FPR reached"] = fpr(g[g.subset == "all"]["testFPR@1pct"])
        rows.append(row)

    for src in BF:
        best = f"E-GraphSAGE ({sel[slug(src)]})"
        e = EGS[(EGS.source == src) & (EGS.protocol == MAIN)]
        add(src, "graph", "HistGB (flow, no graph)", "reference, 30 seeds", e[e.model == "HistGB (flow, no graph)"],
            "–", None, None)
        add(src, "graph", best + ", endpoint chosen on validation", "inductive, 30 seeds",
            e[(e.model == best) & (e["mode"] == "inductive")], "vs HistGB (F8)",
            test_row("F8", "(i)", source=src, candidate=best), test_row("F8", "(iii)", source=src, candidate=best))
        for mode in ("inductive", "transductive"):
            for ep in ("ip", "ip_port", "random"):
                m = f"E-GraphSAGE ({ep})"
                g = e[(e.model == m) & (e["mode"] == mode) & (e.seed < 10)]
                add(src, "graph", m + (" (control)" if ep == "random" else ""), f"{mode}, 10 seeds", g, "vs HistGB (F6)",
                    test_row("F6", "(i)", source=src, protocol=MAIN, mode=mode, candidate=m),
                    test_row("F6", "(iii)", source=src, protocol=MAIN, mode=mode, candidate=m))
        gg = GAT[GAT.source == src]
        for val in ("split validation", "reused validation"):
            for rep_ in (["family-gated", "flow", "flow + neighbourhood", "flow + host", "ensemble (mean of experts)"]
                         if val == "split validation" else ["family-gated"]):
                g = gg[(gg.validation == val) & (gg.representation == rep_)]
                cmp_ = "alternative − gated (F5)" if rep_ != "family-gated" else "–"
                add(src, "gated", rep_, f"{val}, 30 seeds", g, cmp_,
                    test_row("F5", "(i)", source=src, validation=val, candidate=rep_) if rep_ != "family-gated" else None,
                    test_row("F5", "(iii)", source=src, validation=val, candidate=rep_) if rep_ != "family-gated" else None)
    save("M6_model_comparison", "Model comparison at the main protocol: E-GraphSAGE vs flow-only HistGB (families F6, F8) "
         "and the family-gated detector vs each alternative (F5).", pd.DataFrame(rows),
         "Gated rows under split validation set every candidate's threshold on half of validation, so their flow row "
         "differs from Table 5. The gated detector's worst-seed FPR on ToN-IoT (28.7%) is inherited from the host-feature "
         "expert. Random-row E-GraphSAGE runs and per-family values: Tables A3-A6.")


def t7_gated():
    rows = []
    for (src, val, rep), g in GAT.groupby(["source", "validation", "representation"]):
        row = {"source": src, "validation": val, "detector": rep, "seeds": g.seed.nunique()}
        for sub in ("cross-episode", "all"):
            h = g[g.subset == sub]
            row[f"TPR {SUB[sub]}"] = ci(h["TPR@FPR1pct"])
            t = test_row("F5", SUB[sub], source=src, validation=val, candidate=rep) if rep != "family-gated" else None
            row[f"alternative − gated {SUB[sub]}"] = diff(t) if t is not None else "–"
            row[f"p {SUB[sub]}"] = pval(t.p_holm) if t is not None else "–"
        row["FPR reached"] = fpr(g[g.subset == "all"]["testFPR@1pct"])
        rows.append(row)
    save("T7_gated_detector", "Family-gated detector vs each alternative, 30 seeds (family F5). Split validation is "
         "primary: one half of validation chooses experts, the other sets thresholds.", pd.DataFrame(rows),
         "Thresholds in the split-validation rows come from half of validation, so the flow row differs from Table 5.")
    reps = ["flow", "flow + neighbourhood", "flow + host", "ensemble (mean of experts)", "family-gated"]
    fam = per_family("gated_final_gated", BF, MAIN, reps, validation="split validation", label_map={r: r for r in reps})
    save("T7b_gated_by_family", "Per-family TPR at 1% FPR behind Table A5 (split validation, seeds 0-9).", fam)


def t8_sensitivity():
    rows = []
    for src in BF:
        for proto in [MAIN, "episodes gap 1 min, embargo 0 min", "episodes gap 5 min, embargo 5 min",
                      "chronological, embargo 5 min"]:
            g = CTX[(CTX.source == src) & (CTX.protocol == proto) & (CTX.model == "HistGB")
                    & (CTX.representation == "flow") & (CTX.seed < 10)]
            row = {"source": src, "protocol": PROTO_SHORT[proto], "seeds": g[g.subset == "all"].seed.nunique()}
            for sub in ("cross-episode", "all"):
                h = g[g.subset == sub]
                row[f"TPR {SUB[sub]}"] = (ci(h["TPR@FPR1pct"]) + (f" ({h.seed.nunique()} seeds)" if
                                          h.seed.nunique() < row["seeds"] else "")) if len(h) else "not defined"
                row[f"test episodes {SUB[sub]}"] = (fmt(h.test_episodes.mean(), 1) if len(h) and h.test_episodes.mean() > 0
                                                    else "– (split by time)" if len(h) else "–")
            t = test_row("F4", "(iii)", source=src, candidate=proto) if proto != MAIN else None
            row["Δ vs main (iii)"] = diff(t) if t is not None else "–"
            row["p (iii)"] = pval(t.p_holm) if t is not None else "–"
            row["FPR reached"] = fpr(g[g.subset == "all"]["testFPR@1pct"])
            rows.append(row)
    save("T8_protocol_sensitivity", "Protocol sensitivity and drift (finding d): HistGB on flow features, seeds 0-9 "
         "(family F4, compared on subset (iii)).", pd.DataFrame(rows),
         "The chronological split trains on the early part of every class and tests on the later part.")
    fam = per_family("context_final_ctx_main", BF, MAIN, ["HistGB|flow"], subsets=("(iii) all",),
                     label_map={"HistGB|flow": "episodes (main)"})
    ch = per_family("context_final_ctx_chrono", BF, "chronological, embargo 5 min", ["HistGB|flow"], subsets=("(iii) all",),
                    label_map={"HistGB|flow": "chronological"})
    m = fam.merge(ch[["source", "family", "chronological"]], on=["source", "family"], how="outer").drop(
        columns=["subset", "seeds in subset"])
    save("T8b_drift_by_family", "Per-family TPR at 1% FPR, main protocol vs chronological split (HistGB, flow, all test "
         "attacks, seeds 0-9).", m)


def t9_scope():
    r = CTX[(CTX.protocol == MAIN) & (CTX.model == "HistGB") & (CTX.representation == "flow") & (CTX.subset == "all")]
    rows = []
    for src, g in r.groupby("source"):
        if src not in BF:
            continue
        n = g.seed.nunique()
        cnt = {}
        for x in g.dropped_from_cross.fillna(""):
            for f in filter(None, x.split(";")):
                cnt[f] = cnt.get(f, 0) + 1
        for f, c in sorted(cnt.items(), key=lambda kv: -kv[1]):
            eps = int(((BOUNDS.source == src) & (BOUNDS.attack == f)).sum())
            flows = int(AMB_F[(AMB_F.source == src) & (AMB_F.attack == f)].flows.squeeze())
            reason = ("fewer than 3 full-data episodes (split by time)" if eps < 3 else
                      "fewer than 500 flows in the class" if flows < 500 else "fewer than 3 test episodes")
            rows.append({"source": src, "family": f, "full-data episodes": eps, "flows": f"{flows:,}",
                         "seeds excluded from (i)": f"{c}/{n}", "reason": reason})
    save("T9a_within_episode_families", "Families that can only be tested within their own episode (finding e), main "
         "protocol, 30 seeds.", pd.DataFrame(rows))
    rows = []
    for src in BF:
        g = CTX[(CTX.source == src) & (CTX.protocol == MAIN) & (CTX.model == "HistGB")
                & (CTX.representation == "flow") & (CTX.subset == "cross-episode")]
        rows.append({"source": src, "seeds": g.seed.nunique(), "TPR (i), seed-bootstrap CI": ci(g["TPR@FPR1pct"]),
                     "test episodes per seed (median)": fmt(g.n_episodes.median(), 0),
                     "episode-bootstrap CI within a seed (median bounds)":
                         f"[{g['TPR@FPR1pct_ci_low'].median():.3f}, {g['TPR@FPR1pct_ci_high'].median():.3f}]",
                     "FPR reached": fpr(g["testFPR@1pct"])})
    save("T9b_episode_uncertainty", "Uncertainty from the small number of test episodes (finding e): seed-level vs "
         "episode-level 95% CI for HistGB on flow features, subset (i).", pd.DataFrame(rows))


def t10_external():
    rows = []
    for kind, frame in (("splits", SPL), ("context", CTX)):
        f = frame[frame.dataset.isin(["unsw", "ciciot"])]
        for (src, proto, model, rep), g in f.groupby(["source", "protocol", "model", "representation"]):
            row = {"dataset": src, "experiment": kind, "split": PROTO_SHORT[proto], "model": model, "features": rep,
                   "seeds": g.seed.nunique()}
            part = {"splits": "splits_final_unsw_splits" if "UNSW" in src else "splits_final_ciciot_splits",
                    "context": "context_final_unsw_context" if "UNSW" in src else "context_final_ciciot_context"}[kind]
            n_tied = tied(part, src, proto, f"{model}|{rep}")
            for sub in ("cross-episode", "all"):
                h = g[g.subset == sub]
                row[f"TPR {SUB[sub]}"] = tpr_cell(h["TPR@FPR1pct"], n_tied) if len(h) else "–"
            row["FPR reached"] = fpr(g[g.subset == "all"]["testFPR@1pct"])
            rows.append(row)
    save("T10_external", "External validation: original UNSW-NB15 (full protocol, main gap) and CIC-IoT2023 (random "
         "split only: no timestamps or addresses).", pd.DataFrame(rows))
    fam = per_family("context_final_unsw_context", ["UNSW-NB15 (original)"], MAIN,
                     ["HistGB|flow", "HistGB|flow + host"], label_map={"HistGB|flow": "flow", "HistGB|flow + host": "flow + host"})
    save("T10b_external_by_family", "Per-family TPR at 1% FPR behind Table A11 (UNSW-NB15, HistGB, main protocol).", fam)


def t11_old_new():
    o = pd.read_csv(STATS / "old_vs_new.csv")
    i = o[o.new_subset == "(i) cross-episode"][["kind", "source", "protocol", "model", "representation", "TPR_new"]]
    o = o[o.new_subset == "(iii) all"].merge(i, on=["kind", "source", "protocol", "model", "representation"],
                                              how="left", suffixes=("", " (i)"))
    d = pd.DataFrame({"experiment": o.kind, "source": o.source, "split": o.protocol, "model": o.model,
                      "features": o.representation, "v1 TPR": o.TPR_v1.map(fmt), "final TPR (iii)": o.TPR_new.map(fmt),
                      "final TPR (i)": o["TPR_new (i)"].map(fmt), "change (iii)": o.TPR_change.map(lambda v: f"{v:+.3f}")})
    save("T11_old_vs_new", "Earlier draft (v1: 60k-flow subsample, look-ahead host features, transductive context, "
         "different splits) vs final run (seeds 0-9). Descriptive only: seeds are not paired.", d)


def t12_stats():
    lines = (F / "reports/stats_protocol.md").read_text().split("## Families of tests")[1].split("## Not reported")[0]
    rows = []
    for ln in [x for x in lines.splitlines() if x.startswith("- **")]:
        name = ln.split("**")[1]
        size = TESTS[TESTS.family == name].family_size.max() if name in set(TESTS.family) else np.nan
        tests = int((TESTS.family == name).sum()) if name in set(TESTS.family) else 0
        rows.append({"family": name, "tests run (both metrics, both subsets)": tests,
                     "tests per Holm family": "" if np.isnan(size) else int(size), "definition": ln.split("**", 2)[2].strip(" —")})
    save("T12_statistics_protocol", "Declared test families (committed before any statistic was computed).",
         pd.DataFrame(rows))


# ------------------------------------------------------------------------------------- exploratory

def s_tables():
    s = pd.read_csv(STATS / "posthoc_same_family_standardised.csv")
    rows = []
    for src, g in s.groupby("source"):
        d = g["random_same_fam_episode_mix"] - g["episode_(i)"]
        lo, hi = bootstrap_ci(d.to_numpy())
        rows.append({"source": src.replace("_", "-"), "episode split (i)": ci(g["episode_(i)"]),
                     "random rows, same families": ci(g["random_same_fam_own_mix"]),
                     "random rows, same families, episode-split family mix": ci(g["random_same_fam_episode_mix"]),
                     "unseen-episode effect at equal mix": f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"})
    save("S1_same_family", "EXPLORATORY (post-hoc, not pre-specified): subset (i) families scored under the random "
         "split of the same seed. No detectable penalty for unseen episodes at this number of episodes "
         "(HistGB, flow, seeds 0-9).", pd.DataFrame(rows), exploratory=True)
    o = pd.read_csv(STATS / "posthoc_split_decomposition_orders.csv")
    o["value"] = [f"{m * (100 if met == 'F1' else 1):+.{1 if met == 'F1' else 3}f} [{lo * (100 if met == 'F1' else 1):+.{1 if met == 'F1' else 3}f}, "
                  f"{hi * (100 if met == 'F1' else 1):+.{1 if met == 'F1' else 3}f}]"
                  for m, lo, hi, met in zip(o["mean"], o.ci_low, o.ci_high, o.metric)]
    w = o.pivot_table(index=["source", "model", "metric"], columns="term", values="value", aggfunc="first").reset_index()
    w = w[["source", "model", "metric", "total", "mix (mix first)", "mix (threshold first)", "mix (average)",
           "threshold (mix first)", "threshold (threshold first)", "threshold (average)", "remainder"]]
    w["metric"] = w.metric.map({"F1": "weighted F1 (points)", "TPR": "TPR@1%FPR"})
    save("S2_split_gap_decomposition", "EXPLORATORY (post-hoc): random-minus-episode gap (subset iii) split into family "
         "mix, threshold transfer and remainder, in both orders and their average (seeds 0-9).", w, exploratory=True)
    t = pd.read_csv(STATS / "tie_sensitivity_ge.csv")
    g = t.groupby(["part", "source", "protocol", "score"]).agg(seeds=("seed", "nunique"), tied=("tied", "sum"),
                                                                **{c: (c, "mean") for c in t.columns if c.startswith(("TPR", "FPR"))}).reset_index()
    g = g.rename(columns={"FPR_test >": "FPR reached (>)", "FPR_test >=": "FPR reached (≥)"})
    for c in g.columns:
        if g[c].dtype == float:
            g[c] = g[c].map(fmt)
    save("S3_tie_sensitivity", "EXPLORATORY (post-hoc deviation): the 11 cells with tied thresholds, frozen rule "
         "(score > τ) vs score ≥ τ, with the FPR each rule reaches on test.", g, exploratory=True)


def s4_ambiguity():
    d = pd.read_csv(STATS / "posthoc_ambiguity_removed.csv")
    rows = []
    for (src, feat), g in d.groupby(["source", "features"]):
        diff_ = (g["TPR_(i)_benign_identical_removed"] - g["TPR_(i)_all"]).to_numpy()
        lo, hi = bootstrap_ci(diff_)
        rows.append({"source": src, "features": feat, "seeds": g.seed.nunique(),
                     "TPR (i), all attack flows": ci(g["TPR_(i)_all"]),
                     "TPR (i), benign-identical flows removed": ci(g["TPR_(i)_benign_identical_removed"]),
                     "change": f"{diff_.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]",
                     "share of (i) weight removed": fmt(g["removed_weight_share"].mean()),
                     "FPR reached": fpr(g["FPR_reached"])})
    save("S4_ambiguity_removed", "EXPLORATORY (post-hoc deviation): subset (i) TPR at 1% FPR with test attack flows "
         "whose exact behaviour vector also occurs as benign in the full data removed (HistGB, main protocol, seeds "
         "0-9). Threshold unchanged.", pd.DataFrame(rows), exploratory=True)


def preview():
    heads = {"main": "Main-text tables (frozen protocol)", "supplement": "Supplementary tables (frozen protocol)",
             "exploratory": "Exploratory tables (post-hoc analyses, labelled as deviations; not pre-specified)"}
    lines = ["# Paper 2 tables (preview)\n",
             "Generated by `scripts/make_paper2_tables.py` from the committed final-run files. Every table also exists "
             "as CSV and LaTeX in `main/`, `supplement/` and `exploratory/`. Response-time metrics are not reported here "
             "(pre-specified in PROTOCOL_FREEZE.md, reported in paper 3).\n"]
    for folder in ("main", "supplement", "exploratory"):
        lines.append(f"# {heads[folder]}\n")
        for f, name, title, note, frame, expl in [t for t in TABLES if t[0] == folder]:
            lines += [f"## {name}\n", f"*{title}*" + (f" {note}" if note else "") + "\n"]
            lines += ["| " + " | ".join(frame.columns) + " |", "|" + "|".join("---" for _ in frame.columns) + "|"]
            lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in frame.itertuples(index=False)]
            lines.append("")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    for f in (t1_data, t2_composition, t3_thresholds, t4_splits, t5_context, t6_egraphsage, t7_gated,
              t8_sensitivity, t9_scope, t10_external, t11_old_new, t12_stats, s_tables, m6_models, s4_ambiguity):
        f()
    preview()
    print("tables:", [f"{t[0]}/{t[1]}" for t in TABLES])


if __name__ == "__main__":
    main()

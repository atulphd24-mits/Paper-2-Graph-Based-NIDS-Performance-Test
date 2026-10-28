"""
Statistics for the paper 2 final run (frozen protocol, PROTOCOL_FREEZE.md section 6).

Test families are declared in FAMILIES below and written to
outputs/paper2_final/reports/stats_protocol.md with `--declare`, which is committed BEFORE any
statistic is computed. Without `--declare` the script also runs every test:

- paired over seeds: Wilcoxon signed-rank (two-sided), Holm within each family, paired bootstrap
  95% CI of the mean difference (10,000 resamples) -- src.stats.protocol.compare;
- on subset (i) cross-episode (primary) and subset (iii) all, as separate families; a comparison
  is run on (i) only if every seed of both conditions has subset (i) rows;
- metrics: TPR@FPR1pct (primary) and PR_AUC_w (secondary), separate families.

Also writes descriptive tables (mean and seed-bootstrap CI per cell, FPR reached, test episodes,
median within-seed episode-bootstrap CI) and the descriptive old (v1) vs new comparison.

Only the final-run files (*_final_*.csv) are read; timing, check and smoke CSVs are never used.

Usage:
    python scripts/paper2_stats.py --declare   # write the family declaration only
    python scripts/paper2_stats.py             # run everything
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stats.protocol import bootstrap_ci, compare, holm  # noqa: E402

RES = ROOT / "outputs/paper2_final/results"
STATS = ROOT / "outputs/paper2_final/stats"
REPORTS = ROOT / "outputs/paper2_final/reports"
V1 = ROOT / "outputs/paper2_final/v1_baseline/leakage_study"
METRICS = ["TPR@FPR1pct", "PR_AUC_w"]
SUBSETS = {"(i) cross-episode": "cross-episode", "(iii) all": "all"}
MAIN = "episodes gap 1 min, embargo 5 min"
BASE = "HistGB (flow, no graph)"

FILES = {
    "splits": ["splits_raw_final_splits.csv", "splits_raw_final_unsw_splits.csv", "splits_raw_final_ciciot_splits.csv"],
    "context": ["context_raw_final_ctx_main.csv", "context_raw_final_hl_ctx_main.csv", "context_raw_final_ctx_gap5.csv",
                "context_raw_final_ctx_chrono.csv", "context_raw_final_ctx_e0.csv", "context_raw_final_unsw_context.csv",
                "context_raw_final_ciciot_context.csv"],
    "gated": ["gated_raw_final_gated.csv", "gated_raw_final_hl_gated.csv"],
    "egraphsage": ["egraphsage_raw_final_egs_ind.csv", "egraphsage_raw_final_egs_tr.csv",
                   "egraphsage_raw_final_hl_egs_ind_ToN_IoT.csv", "egraphsage_raw_final_hl_egs_ind_CSE_CIC_IDS2018.csv",
                   "egraphsage_raw_final_unsw_egs.csv"],
}

# name: (description, experiment, condition column, baseline, group keys, row filter)
FAMILIES = {
    "F1 split inflation": (
        "Random rows and the chronological split vs the main episode protocol, per (source, model, "
        "representation incl. timestamp-only): 2 tests per family. Subset (iii) only: subset (i) is not "
        "defined for random rows or the chronological split. 10 seeds.",
        "splits", "protocol", MAIN, ["source", "model", "representation"], None),
    "F2 context features (main protocol)": (
        "flow + neighbourhood, flow + host, flow + neighbourhood + host vs flow, per (source, model): "
        "3 tests per family. HistGB 30 seeds, RF and MLP 10 seeds; external datasets 10 seeds "
        "(CIC-IoT2023: flow + neighbourhood only, random split, subset (iii)).",
        "context", "representation", "flow", ["source", "model", "protocol"],
        lambda d: d[d.protocol.isin([MAIN, "random rows"])]),
    "F3 context features (sensitivity protocols)": (
        "Same comparison as F2 for HistGB under the 5-min gap, the chronological reference and no "
        "embargo (E = 0), per (source, protocol): 3 tests per family. 10 seeds.",
        "context", "representation", "flow", ["source", "model", "protocol"],
        lambda d: d[~d.protocol.isin([MAIN, "random rows"])]),
    "F4 protocol sensitivity (flow features)": (
        "HistGB on flow features: 5-min gap, chronological reference and E = 0 vs the main protocol, "
        "per source: 3 tests per family. Subset (iii) (subset (i) is not available for every seed "
        "of the 5-min gap and not defined for the chronological split). 10 seeds (seeds 0-9 of the "
        "main protocol).",
        "context", "protocol", MAIN, ["source", "model", "representation"],
        lambda d: d[(d.model == "HistGB") & (d.representation == "flow") & (d.seed < 10)
                    & d.source.isin(["ToN-IoT", "CSE-CIC-IDS2018"])]),
    "F5 family-gated detector": (
        "Each alternative (flow, flow + neighbourhood, flow + host, unweighted ensemble) vs the "
        "family-gated detector, per (source, validation version): 4 tests per family; reported "
        "differences are alternative minus gated. 30 seeds. Primary version: split validation.",
        "gated", "representation", "family-gated", ["source", "validation"], None),
    "F6 E-GraphSAGE vs flow-only HistGB": (
        "Each endpoint (IP, IP:port, random control) vs flow-only HistGB on the same split, per "
        "(source, protocol, training mode): 3 tests per family. Seeds 0-9.",
        "egraphsage", "model", BASE, ["source", "protocol", "mode"], None),
    "F7 inductive vs transductive E-GraphSAGE": (
        "Inductive minus transductive, per endpoint, per (source, protocol): 3 tests per family. "
        "Seeds 0-9.", "egraphsage", "mode", "transductive", ["source", "protocol", "model"], None),
    "F8 headline E-GraphSAGE (30 seeds)": (
        "Inductive E-GraphSAGE with the endpoint chosen on validation PR-AUC "
        "(reports/endpoint_selection.json) vs flow-only HistGB, main protocol, per source: 1 test "
        "per family. 30 seeds.", "egraphsage", "model", BASE, ["source", "protocol"], None),
}


def load(kind):
    frames = []
    for name in FILES[kind]:
        path = RES / name
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path).assign(file=name))
    d = pd.concat(frames, ignore_index=True)
    keys = [c for c in ("dataset", "source", "seed", "protocol", "model", "representation", "validation",
                        "mode", "subset") if c in d.columns]
    dup = d.duplicated(keys).sum()
    if dup:
        raise ValueError(f"{kind}: {dup} duplicated result rows")
    return d


def run_tests(name, frame, cond, baseline, keys, candidates=None):
    out = []
    for label, subset in SUBSETS.items():
        sub = frame[frame.subset == subset]
        if sub.empty:
            continue
        # (i) only where every seed of every condition in the group has subset (i) rows
        if subset == "cross-episode":
            full = frame[frame.subset == "all"].groupby(keys + [cond]).seed.nunique()
            have = sub.groupby(keys + [cond]).seed.nunique()
            ok = (have.reindex(full.index) == full).groupby(level=list(range(len(keys)))).all()
            ok = ok[ok].index
            sub = sub.set_index(keys).loc[sub.set_index(keys).index.isin(ok)].reset_index()
            if sub.empty:
                continue
        res = compare(sub, cond, baseline, METRICS, keys, candidates=candidates)
        out.append(res.assign(family=name, subset=label))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def holm_f7(tests):
    """
    F7 is declared as 3 tests per family (the endpoints within one source and protocol), but compare()
    groups by model, which left one test per family. Re-apply Holm over the endpoints within each
    (source, protocol, metric, subset), as declared (correction made after results were seen).
    """
    f7 = tests.family.str.startswith("F7")
    g = tests[f7].groupby(["source", "protocol", "metric", "subset"])
    tests.loc[f7, "family_size"] = g["p_raw"].transform("size")
    tests.loc[f7, "p_holm"] = g["p_raw"].transform(lambda p: pd.Series(holm(p.to_numpy()), index=p.index))
    return tests


def descriptive(frame, keys):
    rows = []
    for k, g in frame.groupby(keys + ["subset"], dropna=False):
        row = dict(zip(keys + ["subset"], k if isinstance(k, tuple) else (k,)), seeds=g.seed.nunique())
        for m in METRICS + ["TPR@FPR0.1pct"]:
            v = g[m].to_numpy(float)
            lo, hi = bootstrap_ci(v)
            row.update({f"{m}_mean": v.mean(), f"{m}_ci_low": lo, f"{m}_ci_high": hi})
        row["testFPR@1pct_mean"] = g["testFPR@1pct"].mean()
        row["testFPR@1pct_max"] = g["testFPR@1pct"].max()
        row["test_episodes_mean"] = g["test_episodes"].mean() if "test_episodes" in g else np.nan
        if "TPR@FPR1pct_ci_low" in g and g["TPR@FPR1pct_ci_low"].notna().any():
            row["episode_ci_low_median"] = g["TPR@FPR1pct_ci_low"].median()
            row["episode_ci_high_median"] = g["TPR@FPR1pct_ci_high"].median()
            row["n_episodes_median"] = g["n_episodes"].median()
        rows.append(row)
    return pd.DataFrame(rows)


def egs_frame(eg):
    """Share each seed's flow-only HistGB baseline with every training mode (same split)."""
    base = eg[eg.model == BASE]
    parts = [eg[eg["mode"] == m] for m in ("inductive", "transductive")]
    parts += [base.assign(mode=m) for m in ("inductive", "transductive")]
    d = pd.concat(parts, ignore_index=True)
    return d.drop_duplicates(["source", "seed", "protocol", "model", "mode", "subset"])


def old_vs_new(context, splits, gated, eg):
    """Descriptive: v1 (60k subsample, v1 protocol, all test flows) vs new (main protocol)."""
    v1 = []
    for f, model, kind in (("part4_graph_raw.csv", "HistGB", "context"),
                           ("robustA2_features_models_raw.csv", None, "context"),
                           ("part3_splits_raw.csv", "HistGB", "splits"),
                           ("robustA1_splits_models_raw.csv", None, "splits"),
                           ("part5_two_branch_raw.csv", "HistGB", "gated"),
                           ("egraphsage_raw.csv", None, "egraphsage")):
        d = pd.read_csv(V1 / f)
        d = d[d.source.isin(["ToN-IoT", "CSE-CIC-IDS2018"])]
        if model:
            d = d.assign(model=model)
        if "representation" not in d:
            d = d.assign(representation="behaviour" if kind == "splits" else "-")
        d = d.assign(kind=kind, split=d["split"] if "split" in d else "attack episodes")
        d["representation"] = d["representation"].replace({"family-gated (ours)": "family-gated"})
        v1.append(d)
    v1 = pd.concat(v1, ignore_index=True)
    v1 = v1[v1.split.isin(["attack episodes", "random rows"])]
    v1 = v1.assign(protocol=v1.split.map({"attack episodes": "episodes", "random rows": "random rows"}))
    v1.loc[v1.kind == "egraphsage", "mode"] = "transductive"
    old = v1.groupby(["kind", "source", "protocol", "model", "representation"], dropna=False).agg(
        seeds_v1=("seed", "nunique"), TPR_v1=("TPR@FPR1pct", "mean"), PR_AUC_v1=("PR_AUC_w", "mean")).reset_index()

    new_parts = []
    c = context[(context.protocol == MAIN) & context.source.isin(["ToN-IoT", "CSE-CIC-IDS2018"]) & (context.seed < 10)]
    new_parts.append(c.assign(kind="context", protocol="episodes"))
    s = splits[splits.source.isin(["ToN-IoT", "CSE-CIC-IDS2018"]) & splits.protocol.isin([MAIN, "random rows"])]
    new_parts.append(s.assign(kind="splits", protocol=s.protocol.replace({MAIN: "episodes"})))
    gg = gated[(gated.validation == "reused validation") & (gated.seed < 10)]
    new_parts.append(gg.assign(kind="gated", protocol="episodes", model="HistGB"))
    e = eg[(eg["mode"].isin(["transductive", "-"])) & (eg.seed < 10)]
    new_parts.append(e.assign(kind="egraphsage", representation="-",
                              protocol=e.protocol.replace({MAIN: "episodes"})))
    new = pd.concat(new_parts, ignore_index=True)
    rows = []
    for label, subset in SUBSETS.items():
        n = new[new.subset == subset].groupby(["kind", "source", "protocol", "model", "representation"]).agg(
            seeds_new=("seed", "nunique"), TPR_new=("TPR@FPR1pct", "mean"), PR_AUC_new=("PR_AUC_w", "mean"))
        rows.append(old.merge(n.reset_index(), on=["kind", "source", "protocol", "model", "representation"],
                              how="inner").assign(new_subset=label))
    out = pd.concat(rows, ignore_index=True)
    out["TPR_change"] = out.TPR_new - out.TPR_v1
    out["PR_AUC_change"] = out.PR_AUC_new - out.PR_AUC_v1
    return out


def declaration():
    lines = ["# Statistics protocol (paper 2 final)\n",
             "Declared before any statistic of the final run was computed (committed with this file).\n",
             "All model comparisons are **paired by seed**: each seed fixes the split and the model's random "
             "state, and every condition in a comparison uses the same seeds and the same split.\n",
             "For every comparison: `mean_diff` (mean over seeds of candidate − baseline); `ci_low`, `ci_high` "
             "(paired bootstrap 95% CI, 10,000 resamples of seeds, percentile, RNG seed 0); `p_raw` (two-sided "
             "Wilcoxon signed-rank on per-seed differences; not computed with < 5 seeds or all differences 0); "
             "`p_holm` (Holm–Bonferroni within the family); `family_size`; `wins` (seeds where the candidate "
             "is higher).\n",
             "Metrics: `TPR@FPR1pct` (primary) and `PR_AUC_w` (secondary), each its own family. Subsets: "
             "(i) cross-episode (primary) and (iii) all, each its own family; a comparison is tested on (i) "
             "only if every seed of every condition has subset (i) rows. Descriptive tables report means with "
             "seed-bootstrap CIs, the FPR reached on test, test episodes and the median within-seed "
             "episode-bootstrap CI for subset (i).\n",
             "With 10 seeds the smallest two-sided Wilcoxon p is 0.002; with 30 seeds about 1.9e-9.\n",
             "## Families of tests\n"]
    for name, (desc, exp, cond, base, keys, _) in FAMILIES.items():
        lines.append(f"- **{name}** ({exp}; condition `{cond}`, baseline `{base}`, one family per "
                     f"({', '.join(keys)}), metric and subset). {desc}")
    lines += ["- **Old (v1) vs new** — descriptive only, no test: v1 used the 60k-flow subsample, the "
              "leaky host features, transductive neighbourhood context and different splits, so seeds are "
              "not paired. Means per cell; new values on subsets (i) and (iii) of the main protocol, "
              "seeds 0-9.\n",
              "## Not reported in paper 2\n",
              "Response-time metrics (a)–(d) (PROTOCOL_FREEZE.md, section 6) are computed separately into "
              "`outputs/paper3_baseline/` for paper 3.\n"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "stats_protocol.md").write_text(declaration())
    if args.declare:
        print("declaration written")
        return
    STATS.mkdir(parents=True, exist_ok=True)

    splits, context, gated = load("splits"), load("context"), load("gated")
    eg = egs_frame(load("egraphsage"))
    tests = []
    for name, (_, exp, cond, base, keys, filt) in FAMILIES.items():
        frame = {"splits": splits, "context": context, "gated": gated, "egraphsage": eg}[exp]
        frame = filt(frame) if filt else frame
        if name.startswith("F6"):
            frame = frame[frame.seed < 10]
        if name.startswith("F7"):
            frame = frame[(frame.seed < 10) & frame.model.str.startswith("E-GraphSAGE")]
        if name.startswith("F8"):
            frame = frame[(frame.protocol == MAIN) & (frame["mode"] == "inductive")]
            seeds30 = frame[frame.model != BASE].groupby(["source", "model"]).seed.nunique()
            best = seeds30[seeds30 >= 30].reset_index()
            frame = pd.concat([frame[frame.model == BASE]] + [frame[(frame.source == r.source) & (frame.model == r.model)]
                                                              for r in best.itertuples()])
        tests.append(run_tests(name, frame, cond, base, keys))
    tests = pd.concat(tests, ignore_index=True)
    tests = holm_f7(tests)
    tests.to_csv(STATS / "tests_all_families.csv", index=False)

    descriptive(splits, ["dataset", "source", "protocol", "model", "representation"]).to_csv(
        STATS / "splits_descriptive.csv", index=False)
    descriptive(context, ["dataset", "source", "protocol", "model", "representation"]).to_csv(
        STATS / "context_descriptive.csv", index=False)
    descriptive(gated, ["source", "validation", "representation"]).to_csv(STATS / "gated_descriptive.csv", index=False)
    descriptive(eg, ["dataset", "source", "protocol", "mode", "model"]).to_csv(
        STATS / "egraphsage_descriptive.csv", index=False)
    old_vs_new(context, splits, gated, eg).to_csv(STATS / "old_vs_new.csv", index=False)
    print("written:", sorted(p.name for p in STATS.glob("*.csv")))


if __name__ == "__main__":
    main()

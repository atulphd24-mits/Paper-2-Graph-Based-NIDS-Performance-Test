"""
Completeness check of Phase A (seeds 0-9) of the full run. Read-only; reports counts only, never
detection metrics.

1. Result rows: every expected (source, seed, protocol, model/feature set) combination present
   exactly once per evaluated subset.
2. Prediction files: expected number per part; score columns complete, no NaN, not constant; row
   ids identical to the validation + test rows of the split recomputed independently with
   Data.make_split (for the gated split-validation files: test + the threshold half).
3. Run integrity: meta status, lane log end markers, per-seed "done" lines.

Usage: python scripts/check_phase_a.py   -> outputs/paper2_final/reports/phase_a_completeness.md
"""
import json
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from paper2_final import Data, REPS  # noqa: E402
from smoke_compare import md  # noqa: E402
from src.models.evaluation import split_validation  # noqa: E402
from src.models.predictions import slug  # noqa: E402

OUT = ROOT / "outputs/paper2_final"
SEEDS = range(10)
BF = ["ToN-IoT", "CSE-CIC-IDS2018"]
EP1, EP5 = "episodes gap 1 min, embargo 5 min", "episodes gap 5 min, embargo 5 min"
CHR, E0, RND = "chronological, embargo 5 min", "episodes gap 1 min, embargo 0 min", "random rows"
M3 = ["HistGB", "RandomForest", "MLP"]
SPLIT_SETS = [("HistGB", "behaviour"), ("RandomForest", "behaviour"), ("MLP", "behaviour"), ("HistGB", "timestamp only")]
GATED = ["flow", "flow + neighbourhood", "flow + host", "ensemble (mean of experts)", "family-gated"]
ENDP = ["ip", "ip_port", "random"]

# part: (experiment, dataset, sources, protocols, combos [(key columns...)], key names, split args)
PARTS = {
    "splits": ("splits", "bigflow", BF, [RND, EP1, CHR], SPLIT_SETS, ("model", "representation")),
    "ctx_main": ("context", "bigflow", BF, [EP1], list(product(M3, REPS)), ("model", "representation")),
    "ctx_gap5": ("context", "bigflow", BF, [EP5], [("HistGB", r) for r in REPS], ("model", "representation")),
    "ctx_chrono": ("context", "bigflow", BF, [CHR], [("HistGB", r) for r in REPS], ("model", "representation")),
    "ctx_e0": ("context", "bigflow", BF, [E0], [("HistGB", r) for r in REPS], ("model", "representation")),
    "gated": ("gated", "bigflow", BF, [EP1], list(product(["reused validation", "split validation"], GATED)),
              ("validation", "representation")),
    "unsw_splits": ("splits", "unsw", ["UNSW-NB15 (original)"], [RND, EP1, CHR], SPLIT_SETS, ("model", "representation")),
    "unsw_context": ("context", "unsw", ["UNSW-NB15 (original)"], [EP1], list(product(M3, REPS)), ("model", "representation")),
    "ciciot_splits": ("splits", "ciciot", ["CIC-IoT2023"], [RND], SPLIT_SETS[:3], ("model", "representation")),
    "ciciot_context": ("context", "ciciot", ["CIC-IoT2023"], [RND], list(product(M3, ["flow", "flow + neighbourhood"])),
                       ("model", "representation")),
    "egs_ind": ("egraphsage", "bigflow", BF, [EP1, RND],
                [("HistGB (flow, no graph)", "-")] + [(f"E-GraphSAGE ({e})", "inductive") for e in ENDP], ("model", "mode")),
    "egs_tr": ("egraphsage", "bigflow", BF, [EP1, RND], [(f"E-GraphSAGE ({e})", "transductive") for e in ENDP],
               ("model", "mode")),
    "unsw_egs": ("egraphsage", "unsw", ["UNSW-NB15 (original)"], [EP1, RND],
                 [(f"E-GraphSAGE ({e})", m) for m in ("inductive", "transductive") for e in ENDP], ("model", "mode")),
}
UNITS = {"splits": 2, "ctx_main": 2, "ctx_gap5": 2, "ctx_chrono": 2, "ctx_e0": 2, "gated": 2, "unsw_splits": 1,
         "unsw_context": 1, "ciciot_splits": 1, "ciciot_context": 1, "egs_ind": 4, "egs_tr": 4, "unsw_egs": 2}
PROTO_ARGS = {RND: ("random", 0, 0), EP1: ("episodes", 1, 5), EP5: ("episodes", 5, 5), CHR: ("chronological", 0, 5),
              E0: ("episodes", 1, 0)}


def score_name(part, combo):
    exp = PARTS[part][0]
    if exp in ("splits", "context"):
        return f"{combo[0]}|{combo[1]}"
    if exp == "gated":
        return combo[1]
    return combo[0] if combo[1] == "-" else f"{combo[0]}|{combo[1]}"


def check_results(part):
    exp, dataset, sources, protocols, combos, keys = PARTS[part]
    r = pd.read_csv(OUT / "results" / f"{exp}_raw_final_{part}.csv")
    expected = {(s, seed, p) + c for s, seed, p, c in product(sources, SEEDS, protocols, combos)}
    cols = ["source", "seed", "protocol", *keys]
    got = r.groupby(cols + ["subset"]).size()
    dup = int((got > 1).sum())
    present = set(map(tuple, r[cols].drop_duplicates().itertuples(index=False)))
    subsets = r.groupby(cols)["subset"].apply(lambda x: ",".join(sorted(x))).value_counts().to_dict()
    return {"part": part, "result_rows": len(r), "expected_combos": len(expected),
            "missing": len(expected - present), "unexpected": len(present - expected),
            "duplicated (combo, subset)": dup, "subset patterns": "; ".join(f"{k}: {v}" for k, v in subsets.items())}


def check_predictions(part, datas, split_cache):
    exp, dataset, sources, protocols, combos, keys = PARTS[part]
    root = OUT / "predictions" / f"{exp}_final_{part}" / dataset
    variants = ["reused validation", "split validation"] if exp == "gated" else [None]
    files = sorted(root.rglob("*.parquet"))
    n_expected = len(sources) * len(SEEDS) * len(protocols) * len(variants)
    issues = []
    for s, seed, p, v in product(sources, SEEDS, protocols, variants):
        name = f"seed{seed}_{slug(p)}" + (f"_{slug(v)}" if v else "") + ".parquet"
        path = root / slug(s) / name
        if not path.exists():
            issues.append(f"missing {path.name} ({s})")
            continue
        f = pd.read_parquet(path)
        need = [score_name(part, c) for c in combos if v is None or c[0] == v]
        miss = [c for c in need if c not in f.columns]
        if miss:
            issues.append(f"{path.name}: missing score columns {miss}")
        for c in [c for c in need if c in f.columns]:
            if f[c].isna().any():
                issues.append(f"{path.name}: NaN in {c}")
            if f[c].nunique() < 2:
                issues.append(f"{path.name}: constant {c}")
        key = (dataset, s, seed, p)
        if key not in split_cache:
            data = datas[dataset]
            sdf = data.source(s, subsample=False) if exp == "splits" else data.source(s)
            split, episode, mode = data.make_split(sdf, seed, *PROTO_ARGS[p])
            split_cache[key] = (sdf, split, episode, mode)
        sdf, split, episode, mode = split_cache[key]
        if v == "split validation":
            strata = (sdf["source"] + "|" + sdf["attack"]).to_numpy()
            halves = split_validation(split, episode, mode, strata, sdf["flow_start_milliseconds"].to_numpy(),
                                      random_state=seed)
            rows = np.flatnonzero(np.isin(halves, ["validation_threshold", "test"]))
        else:
            rows = np.flatnonzero(np.isin(split, ["validation", "test"]))
        if not np.array_equal(f["row"].to_numpy(), rows):
            issues.append(f"{path.name} ({s}): rows differ from recomputed split ({len(f)} vs {len(rows)})")
        elif not np.array_equal(f.loc[f["split"].astype(str) == "test", "row"].to_numpy(), np.flatnonzero(split == "test")):
            issues.append(f"{path.name} ({s}): test rows differ")
    return {"part": part, "prediction_files": len(files), "expected_files": n_expected,
            "file_issues": len(issues), "examples": "; ".join(issues[:3])}


def check_run(part):
    exp, dataset = PARTS[part][:2]
    meta = json.loads((OUT / "logs" / f"{exp}_{dataset}_final_{part}_meta.json").read_text())
    log = (OUT / "logs" / ("run_paper2_final_graph.log" if part.startswith(("egs", "unsw_egs"))
                           else "run_paper2_final_tabular.log")).read_text(errors="replace")
    block = re.search(rf"=== {part} start.*?=== {part} (end|FAILED)", log, re.S)
    done = len(re.findall(r"seed \d+ done$", block.group(0), re.M)) if block else 0
    return {"part": part, "meta_status": meta["status"], "commit": meta["git_commit"][:7],
            "dirty": meta["git_dirty_tracked"], "seeds": ",".join(map(str, meta["args"]["seeds"])) == ",".join(map(str, SEEDS)),
            "log_end": block.group(1) if block else "none", "done_lines": done, "expected_done": UNITS[part] * 10}


def main():
    datas = {d: Data(d) for d in ("bigflow", "unsw", "ciciot")}
    for d in datas.values():
        d.subsample = False
    cache = {}
    runs = pd.DataFrame([check_run(p) for p in PARTS])
    res = pd.DataFrame([check_results(p) for p in PARTS])
    pred = pd.DataFrame([check_predictions(p, datas, cache) for p in PARTS])
    lines = ["# Phase A completeness check (seeds 0-9)\n",
             "Read-only; counts only, no detection metrics. Splits recomputed independently with "
             "`Data.make_split` and compared with the saved prediction rows.\n",
             "## Run integrity\n", md(runs), "\n## Result rows\n", md(res), "\n## Prediction files\n", md(pred)]
    (OUT / "reports" / "phase_a_completeness.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

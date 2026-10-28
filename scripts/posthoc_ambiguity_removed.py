"""
Post-hoc (DEVIATION, decided 2026-10-21 after results were seen): subset (i) TPR with benign-identical
attack flows removed (exploratory table S4).

A flow is "benign-identical" when its exact behaviour vector also occurs as a benign flow of the same
source anywhere in the full data. Hashing is the v1 label-ambiguity audit's (scripts/label_ambiguity.py):
pandas row hash of the behaviour columns of the raw files, so the flags reproduce its per-family shares.

The alert threshold is unchanged (99th percentile of validation benign scores, score > tau); only test
attack flows flagged as benign-identical are removed before computing the design-weighted TPR.
HistGB, flow and flow + host, main protocol, seeds 0-9.

Usage: python scripts/posthoc_ambiguity_removed.py
Writes outputs/paper2_final/stats/benign_identical_flags.parquet, posthoc_ambiguity_removed.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from paper2_final import Data  # noqa: E402
from src.data.bigflow import behaviour_columns, bigflow_dir, list_files, read_file  # noqa: E402
from src.models.predictions import slug  # noqa: E402
from src.stats.protocol import bootstrap_ci  # noqa: E402

STATS = ROOT / "outputs/paper2_final/stats"
PRED = ROOT / "outputs/paper2_final/predictions/context_final_ctx_main/bigflow"
SOURCES = ["ToN-IoT", "CSE-CIC-IDS2018"]
MAIN = "episodes_gap_1_min_embargo_5_min"


def flags(sample_keys):
    """benign-identical flag for every sampled (file_id, file_row), hashing as the v1 audit does."""
    start = time.time()
    benign = {s: [] for s in SOURCES}
    sampled = []
    columns = None
    for file_id, path in enumerate(list_files(bigflow_dir())):
        df = read_file(path)
        columns = columns or behaviour_columns(df.columns)
        h = pd.util.hash_pandas_object(df[columns], index=False).to_numpy()
        for s in SOURCES:
            m = (df["source"] == s).to_numpy() & (df["attack"] == "Benign").to_numpy()
            if m.any():
                benign[s].append(np.unique(h[m]))
        rows = sample_keys.loc[sample_keys.file_id == file_id, "file_row"].to_numpy()
        if len(rows):
            sampled.append(pd.DataFrame({"file_id": file_id, "file_row": rows, "hash": h[rows],
                                         "source": df["source"].to_numpy()[rows]}))
        print(f"[{time.time() - start:5.0f}s] {Path(path).name}", flush=True)
    benign = {s: np.unique(np.concatenate(v)) for s, v in benign.items()}
    out = pd.concat(sampled, ignore_index=True)
    out["benign_identical"] = False
    for s in SOURCES:
        m = out.source == s
        out.loc[m, "benign_identical"] = np.isin(out.loc[m, "hash"].to_numpy(), benign[s])
    return out.drop(columns="hash")


def main():
    data = Data("bigflow")
    data.subsample = False
    keys = data.df.loc[data.df.source.isin(SOURCES), ["file_id", "file_row"]]
    path = STATS / "benign_identical_flags.parquet"
    if path.exists():
        fl = pd.read_parquet(path)
    else:
        fl = flags(keys)
        fl.to_parquet(path, index=False)

    rows, check = [], []
    for src in SOURCES:
        sdf = data.source(src).merge(fl[["file_id", "file_row", "benign_identical"]], on=["file_id", "file_row"],
                                     how="left", validate="one_to_one")
        ident = sdf["benign_identical"].fillna(False).to_numpy()
        att = sdf["attack"] != "Benign"
        w = sdf["design_weight"]
        for fam, g in sdf[att].groupby("attack"):
            check.append({"source": src, "family": fam,
                          "benign_identical_weighted": np.average(ident[g.index], weights=w[g.index])})
        for seed in range(10):
            f = pd.read_parquet(PRED / slug(src) / f"seed{seed}_{MAIN}.parquet")
            ident_rows = ident[f["row"].to_numpy()]
            for col in ("HistGB|flow", "HistGB|flow + host"):
                s = f[col].to_numpy(float)
                split, y = f["split"].astype(str).to_numpy(), f["label"].to_numpy()
                tau = np.quantile(s[(split == "validation") & (y == 0)], 0.99)
                cross = (split == "test") & (y == 1) & f["in_cross"].to_numpy()
                wt = f["design_weight"].to_numpy(float)
                keep = cross & ~ident_rows
                rows.append({"source": src, "seed": seed, "features": col.split("|")[1],
                             "TPR_(i)_all": np.average(s[cross] > tau, weights=wt[cross]),
                             "TPR_(i)_benign_identical_removed": np.average(s[keep] > tau, weights=wt[keep]),
                             "removed_weight_share": wt[cross & ident_rows].sum() / wt[cross].sum(),
                             "FPR_reached": float((s[(split == "test") & (y == 0)] > tau).mean())})
    d = pd.DataFrame(rows)
    d.to_csv(STATS / "posthoc_ambiguity_removed.csv", index=False)
    c = pd.DataFrame(check)
    c.to_csv(STATS / "benign_identical_check.csv", index=False)
    for (src, feat), g in d.groupby(["source", "features"]):
        out = [f"{src:16} {feat:12}"]
        for k in ("TPR_(i)_all", "TPR_(i)_benign_identical_removed", "removed_weight_share", "FPR_reached"):
            lo, hi = bootstrap_ci(g[k].to_numpy())
            out.append(f"{k}={g[k].mean():.3f}[{lo:.3f},{hi:.3f}]")
        print(" ".join(out))


if __name__ == "__main__":
    main()

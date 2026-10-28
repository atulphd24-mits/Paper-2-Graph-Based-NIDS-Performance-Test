"""
Episode-stratified working samples with design weights and past-only host features.

Attack classes: per (source, class), a 20k cap that keeps min(20, episode size) flows from
every full-data episode (1-min gap) and fills the rest at random
(src/data/stratified_sample.py). Benign flows: the same rows as the v1 working
sample (100k per source). Every row carries ``design_weight`` = 1 / inclusion
probability, so weighted metrics estimate full-data rates.

Outputs (new files):
    data/processed/bigflow_working_sample_strat_causal.parquet
    data/processed/unsw_working_sample_strat_causal.parquet

Usage:
    python scripts/build_stratified_sample.py [--dataset bigflow|unsw|both]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_causal_sample import causal_host_for_source  # noqa: E402
from src.data import external  # noqa: E402
from src.data.bigflow import bigflow_dir, list_files, read_file  # noqa: E402
from src.data.stratified_sample import stratified_episode_sample  # noqa: E402
from src.features.host_causal import CausalHostFeatures  # noqa: E402

P = ROOT / "data/processed"
CAP, K_MIN, GAP = 20_000, 20, 1


def log(message, start=time.time()):
    print(f"[{time.time() - start:6.0f}s] {message}", flush=True)


def episode_index(t, bounds_key):
    """Episode number of each timestamp within one class (full-data bounds)."""
    starts = bounds_key["start_ms"].to_numpy()
    idx = np.searchsorted(starts, t, side="right") - 1
    assert (idx >= 0).all() and (t <= bounds_key["end_ms"].to_numpy()[idx]).all()
    return bounds_key["episode"].to_numpy()[idx]


def select_attacks(frame, bounds, seed=42):
    """frame: key, t, plus row identifiers. Returns selected rows with design_weight."""
    out = []
    for i, (key, g) in enumerate(frame.groupby("key", sort=True)):
        b = bounds[bounds["key"] == key].sort_values("start_ms")
        ep = episode_index(g["t"].to_numpy(), b)
        sel, w = stratified_episode_sample(ep, CAP, K_MIN, np.random.default_rng(seed + i))
        out.append(g.iloc[sel].assign(design_weight=w, episode_full=ep[sel]))
    return pd.concat(out, ignore_index=True)


def build_bigflow():
    bounds = pd.read_parquet(P / "episode_bounds_bigflow.parquet")
    bounds = bounds[bounds["gap_min"] == GAP]
    v1 = pd.read_parquet(P / "bigflow_working_sample.parquet", columns=["file_id", "file_row", "source", "attack"])
    benign_keys = v1[v1["attack"] == "Benign"][["file_id", "file_row", "source"]]

    parts = []
    files = list_files(bigflow_dir())
    for file_id, path in enumerate(files):
        d = read_file(path, ["FLOW_START_MILLISECONDS", "Attack"])
        att = (d["attack"] != "Benign").to_numpy()
        parts.append(pd.DataFrame({"file_id": file_id, "file_row": np.flatnonzero(att),
                                   "key": (d["source"] + "|" + d["attack"]).to_numpy()[att],
                                   "t": d["flow_start_milliseconds"].to_numpy()[att].astype(np.int64)}))
    attacks = pd.concat(parts, ignore_index=True)
    log(f"BigFlow attack flows: {len(attacks):,}")
    chosen = select_attacks(attacks, bounds)
    log(f"Selected attack flows: {len(chosen):,}")

    # Benign: v1 rows, weight = full-data benign count / sampled benign count per source
    src_counts = pd.read_csv(ROOT / "outputs/reports/leakage_study/label_ambiguity_by_source.csv")
    n_benign = {r.source: r.flows * (1 - r.attack_share) for r in src_counts.itertuples()}
    benign_n = benign_keys.groupby("source").size()
    benign = benign_keys.assign(design_weight=benign_keys["source"].map(lambda s: n_benign[s] / benign_n[s]))

    wanted = pd.concat([chosen[["file_id", "file_row", "design_weight", "episode_full"]],
                        benign[["file_id", "file_row", "design_weight"]]], ignore_index=True)
    rows = []
    for file_id, path in enumerate(files):
        w = wanted[wanted["file_id"] == file_id]
        if w.empty:
            continue
        d = read_file(path)
        d["file_id"] = file_id
        d["file_row"] = np.arange(len(d))
        rows.append(d.iloc[w["file_row"].to_numpy()].merge(w, on=["file_id", "file_row"]))
    sample = pd.concat(rows, ignore_index=True)
    log(f"Sample rows: {len(sample):,}")

    host = pd.concat([causal_host_for_source(s, sample.loc[sample["source"] == s, ["file_id", "file_row"]])
                      for s in sorted(sample["source"].unique())], ignore_index=True)
    host.columns = [c if c in ("file_id", "file_row") else f"host_{c}" for c in host.columns]
    sample = sample.merge(host, on=["file_id", "file_row"], how="left")
    assert sample.filter(like="host_").notna().all().all()
    sample = sample.sort_values("flow_start_milliseconds", kind="stable", ignore_index=True)
    sample.to_parquet(P / "bigflow_working_sample_strat_causal.parquet", index=False)
    log(f"Saved BigFlow sample {sample.shape}")


def build_unsw():
    bounds = pd.read_parquet(P / "episode_bounds_unsw.parquet")
    bounds = bounds[bounds["gap_min"] == GAP]
    df = external.load_unsw()
    counts = json.loads((ROOT / "outputs/reports/external/unsw_class_counts.json").read_text())
    # benign: same rows as v1 (same RNG draw as the v1 sampler)
    rate = {a: min(1.0, (100_000 if a == "Benign" else 20_000) / n) for a, n in counts.items()}
    keep_v1 = np.random.default_rng(42).random(len(df)) < df["attack"].map(rate).to_numpy()
    benign_rows = np.flatnonzero(keep_v1 & (df["attack"] == "Benign").to_numpy())

    att = np.flatnonzero((df["attack"] != "Benign").to_numpy())
    frame = pd.DataFrame({"row": att, "key": (df["source"] + "|" + df["attack"]).to_numpy()[att],
                          "t": df["flow_start_milliseconds"].to_numpy()[att].astype(np.int64)})
    chosen = select_attacks(frame, bounds)
    weights = np.zeros(len(df))
    keep = np.zeros(len(df), bool)
    keep[chosen["row"].to_numpy()] = True
    weights[chosen["row"].to_numpy()] = chosen["design_weight"].to_numpy()
    keep[benign_rows] = True
    weights[benign_rows] = counts["Benign"] / len(benign_rows)

    df["_no_flags"] = 0
    host = CausalHostFeatures(flags_column="_no_flags", bytes_column="sbytes", packets_column="spkts",
                              reply_packets_column="dpkts").transform(df, target_mask=keep)
    host.columns = [f"host_{c}" for c in host.columns]
    sample = pd.concat([df.loc[keep].drop(columns="_no_flags").reset_index(drop=True),
                        host.reset_index(drop=True)], axis=1)
    sample["design_weight"] = weights[keep]
    sample = sample.sort_values("flow_start_milliseconds", kind="stable", ignore_index=True)
    sample.to_parquet(P / "unsw_working_sample_strat_causal.parquet", index=False)
    log(f"Saved UNSW-NB15 sample {sample.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bigflow", "unsw", "both"], default="both")
    args = parser.parse_args()
    if args.dataset in ("unsw", "both"):
        build_unsw()
    if args.dataset in ("bigflow", "both"):
        build_bigflow()


if __name__ == "__main__":
    main()

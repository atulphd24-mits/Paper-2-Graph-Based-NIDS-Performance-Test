"""
Working sample with past-only host features (paper 2 final, fix A).

Takes the existing BigFlow working sample (same rows), drops its tumbling-window
host features and recomputes them with ``CausalHostFeatures`` over ALL flows of
each source dataset (a flow only ever sees flows that start no later than it).

Outputs (new files; the original samples are not modified):
    data/processed/bigflow_working_sample_causal.parquet
    data/processed/unsw_working_sample_causal.parquet

Stable order for same-timestamp flows: the stream order of the full data
(BigFlow: file number, then row in file; UNSW-NB15: file order of the 4 raw CSVs).

Usage:
    python scripts/build_causal_sample.py [--dataset bigflow|unsw|both]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bigflow import bigflow_dir, list_files, read_file  # noqa: E402
from src.features.host_causal import CausalHostFeatures  # noqa: E402

SOURCE_SAMPLE = ROOT / "data/processed/bigflow_working_sample.parquet"
OUTPUT = ROOT / "data/processed/bigflow_working_sample_causal.parquet"
HOST_COLUMNS = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT",
                "CLIENT_TCP_FLAGS", "IN_BYTES", "IN_PKTS", "OUT_PKTS"]


def log(message, start=time.time()):
    print(f"[{time.time() - start:6.0f}s] {message}", flush=True)


def causal_host_for_source(source, keys):
    parts = []
    for file_id, path in enumerate(list_files(bigflow_dir())):
        df = read_file(path, HOST_COLUMNS)
        mask = (df["source"] == source).to_numpy()
        df = df.loc[mask].drop(columns="source")
        df["file_id"] = file_id
        df["file_row"] = np.flatnonzero(mask)
        parts.append(df)
    traffic = pd.concat(parts, ignore_index=True)
    log(f"{source}: {len(traffic):,} flows")

    wanted = pd.MultiIndex.from_frame(keys[["file_id", "file_row"]])
    have = pd.MultiIndex.from_frame(traffic[["file_id", "file_row"]])
    target = have.isin(wanted)
    features = CausalHostFeatures().transform(traffic, target_mask=target)
    features["file_id"] = traffic.loc[target, "file_id"].to_numpy()
    features["file_row"] = traffic.loc[target, "file_row"].to_numpy()
    log(f"{source}: features for {target.sum():,} sampled flows")
    return features


def build_unsw():
    """Same rows as data/processed/unsw_working_sample.parquet, past-only host features."""
    import json
    from src.data import external
    old = pd.read_parquet(ROOT / "data/processed/unsw_working_sample.parquet")
    df = external.load_unsw()
    counts = json.loads((ROOT / "outputs/reports/external/unsw_class_counts.json").read_text())
    rate = {a: min(1.0, (100_000 if a == "Benign" else 20_000) / n) for a, n in counts.items()}
    keep = np.random.default_rng(42).random(len(df)) < df["attack"].map(rate).to_numpy()
    base = df.loc[keep].reset_index(drop=True)
    check = ["flow_start_milliseconds", "ipv4_src_addr", "sbytes", "attack"]
    assert len(base) == len(old) and base[check].equals(old[check]), "UNSW rows do not match the v1 sample"
    log(f"UNSW-NB15: {len(df):,} flows, {keep.sum():,} sampled (rows match v1 sample)")
    df["_no_flags"] = 0
    host = CausalHostFeatures(flags_column="_no_flags", bytes_column="sbytes", packets_column="spkts",
                              reply_packets_column="dpkts").transform(df, target_mask=keep)
    host.columns = [f"host_{c}" for c in host.columns]
    out = pd.concat([old.drop(columns=[c for c in old.columns if c.startswith("host_")]),
                     host.reset_index(drop=True)], axis=1)
    assert list(out.filter(like="host_").columns) == [c for c in old.columns if c.startswith("host_")]
    out = out[old.columns]
    path = ROOT / "data/processed/unsw_working_sample_causal.parquet"
    out.to_parquet(path, index=False)
    changed = (out.filter(like="host_").to_numpy() != old.filter(like="host_").to_numpy()).mean()
    log(f"Saved {path} {out.shape}; share of host-feature values that changed: {changed:.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bigflow", "unsw", "both"], default="both")
    args = parser.parse_args()
    if args.dataset in ("unsw", "both"):
        build_unsw()
    if args.dataset in ("bigflow", "both"):
        build_bigflow()


def build_bigflow():
    sample = pd.read_parquet(SOURCE_SAMPLE)
    old_host = [c for c in sample.columns if c.startswith("host_")]
    base = sample.drop(columns=old_host)
    parts = []
    for source in sorted(base["source"].unique()):
        keys = base.loc[base["source"] == source, ["file_id", "file_row"]]
        parts.append(causal_host_for_source(source, keys))
    host = pd.concat(parts, ignore_index=True)
    host.columns = [c if c in ("file_id", "file_row") else f"host_{c}" for c in host.columns]

    out = base.merge(host, on=["file_id", "file_row"], how="left")
    assert len(out) == len(sample), "row count changed"
    assert out.filter(like="host_").notna().all().all(), "missing host features"
    assert list(out.filter(like="host_").columns) == old_host, "host columns differ"
    # Same row order as the original sample.
    out = out.set_index(["file_id", "file_row"]).loc[
        pd.MultiIndex.from_frame(sample[["file_id", "file_row"]])].reset_index()
    out = out[sample.columns]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT, index=False)

    # How much did the features change?
    diff = (out[old_host].to_numpy() != sample[old_host].to_numpy()).mean()
    log(f"Saved {OUTPUT} {out.shape}; share of host-feature values that changed: {diff:.3f}")


if __name__ == "__main__":
    main()

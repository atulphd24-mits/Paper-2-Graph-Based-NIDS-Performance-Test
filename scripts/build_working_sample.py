"""
Build the source-aware working sample used by the leakage study.

1. Sample every file, capped per (source, attack) class.
2. For each source dataset, compute multi-scale host behaviour features on
   ALL of that source's traffic (a deployed sensor sees every flow), then
   attach them to the sampled rows.

Output: data/processed/bigflow_working_sample.parquet

Usage:
    python scripts/build_working_sample.py [--attack-cap 20000] [--benign-cap 100000]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bigflow import bigflow_dir, build_working_sample, list_files, read_file  # noqa: E402
from src.graph.host_features import HostBehaviorFeatures  # noqa: E402

DATA_DIR = bigflow_dir()
OUTPUT = PROJECT_ROOT / "data/processed/bigflow_working_sample.parquet"

HOST_COLUMNS = [
    "FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT",
    "CLIENT_TCP_FLAGS", "IN_BYTES", "IN_PKTS", "OUT_PKTS",
]


def log(message, start=time.time()):
    print(f"[{time.time() - start:6.0f}s] {message}", flush=True)


def host_features_for_source(source, sample_keys):
    """Host features over all flows of one source, returned for sampled rows."""
    parts = []
    for file_id, path in enumerate(list_files(DATA_DIR)):
        df = read_file(path, HOST_COLUMNS)
        mask = df["source"] == source
        df = df.loc[mask].drop(columns="source")
        df["file_id"] = file_id
        df["file_row"] = np.flatnonzero(mask.to_numpy())
        for col in ("ipv4_src_addr", "ipv4_dst_addr"):
            df[col] = df[col].astype("category")
        parts.append(df)

    traffic = pd.concat(parts, ignore_index=True)
    log(f"{source}: {len(traffic):,} flows for host features")

    features = HostBehaviorFeatures().transform(traffic)
    features["file_id"] = traffic["file_id"].to_numpy()
    features["file_row"] = traffic["file_row"].to_numpy()

    keys = sample_keys[["file_id", "file_row"]]
    return keys.merge(features, on=["file_id", "file_row"], how="left")


def main(attack_cap, benign_cap):
    log("Sampling all files")
    sample = build_working_sample(DATA_DIR, attack_cap=attack_cap, benign_cap=benign_cap)
    log(f"Sample: {len(sample):,} rows")
    print(sample.groupby(["source", "attack"]).size().to_string())

    host_parts = []
    for source in sorted(sample["source"].unique()):
        keys = sample.loc[sample["source"] == source, ["file_id", "file_row"]]
        host_parts.append(host_features_for_source(source, keys))
    host = pd.concat(host_parts, ignore_index=True)
    host.columns = [
        c if c in ("file_id", "file_row") else f"host_{c}" for c in host.columns
    ]

    sample = sample.merge(host, on=["file_id", "file_row"], how="left")
    assert sample.filter(like="host_").notna().all().all(), "host features missing"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUTPUT, index=False)
    log(f"Saved {OUTPUT} {sample.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-cap", type=int, default=20_000)
    parser.add_argument("--benign-cap", type=int, default=100_000)
    args = parser.parse_args()
    main(args.attack_cap, args.benign_cap)

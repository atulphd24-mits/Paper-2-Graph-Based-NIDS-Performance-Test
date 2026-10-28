"""
Full-data attack-episode boundaries (paper 2 final, fix 5).

Episodes were previously built from the timestamps of the *sampled* flows, so
sampling stretched the gaps and changed the number of episodes. Here they are
built from the timestamps of every flow of each (source, class), for each gap,
and saved; experiments map their sampled flows onto them (``map_episodes``).

Outputs (new files):
    data/processed/episode_bounds_bigflow.parquet
    data/processed/episode_bounds_unsw.parquet
    outputs/paper2_final/reports/episode_counts_before_after.csv

Usage:
    python scripts/build_episode_bounds.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import external  # noqa: E402
from src.data.bigflow import bigflow_dir, list_files, read_file  # noqa: E402
from src.models.evaluation import episode_bounds  # noqa: E402

GAPS = [1, 5, 30]
OUT = ROOT / "data/processed"
REPORT = ROOT / "outputs/paper2_final/reports/episode_counts_before_after.csv"
V1_COUNTS = ROOT / "outputs/paper2_final/v1_baseline/leakage_study/robustB_gap_episode_counts.csv"


def log(message, start=time.time()):
    print(f"[{time.time() - start:6.0f}s] {message}", flush=True)


def bounds_for(frame):
    """frame: columns source, attack, t (ms). Attack classes only."""
    rows = []
    for (source, attack), g in frame.groupby(["source", "attack"]):
        if attack == "Benign":
            continue
        for gap in GAPS:
            starts, ends, flows = episode_bounds(g["t"].to_numpy(), gap * 60_000)
            for i, (a, b, n) in enumerate(zip(starts, ends, flows)):
                rows.append({"key": f"{source}|{attack}", "source": source, "attack": attack,
                             "gap_min": gap, "episode": i, "start_ms": int(a), "end_ms": int(b),
                             "flows": int(n)})
    return pd.DataFrame(rows)


def main():
    parts = []
    for path in list_files(bigflow_dir()):
        df = read_file(path, ["FLOW_START_MILLISECONDS", "Attack"])
        parts.append(pd.DataFrame({"source": df["source"], "attack": df["attack"],
                                   "t": df["flow_start_milliseconds"].astype("int64")}))
    bigflow = pd.concat(parts, ignore_index=True)
    log(f"BigFlow timestamps: {len(bigflow):,}")
    b = bounds_for(bigflow)
    b.to_parquet(OUT / "episode_bounds_bigflow.parquet", index=False)
    log(f"BigFlow episodes: {len(b):,} rows")

    unsw = external.load_unsw()
    u = bounds_for(pd.DataFrame({"source": unsw["source"], "attack": unsw["attack"],
                                 "t": unsw["flow_start_milliseconds"].astype("int64")}))
    u.to_parquet(OUT / "episode_bounds_unsw.parquet", index=False)
    log(f"UNSW-NB15 episodes: {len(u):,} rows")

    # Before (v1: built on the 60k-flow subsample, seed 0) vs after (full data).
    after = b.groupby(["source", "attack", "gap_min"]).size().rename("episodes_full_data").reset_index()
    before = pd.read_csv(V1_COUNTS).melt(id_vars=["source", "family"], var_name="gap_min",
                                         value_name="episodes_v1_sample")
    before["gap_min"] = before["gap_min"].astype(int)
    before = before.rename(columns={"family": "attack"})
    report = after.merge(before, on=["source", "attack", "gap_min"], how="left")
    report = report[report.source.isin(["ToN-IoT", "CSE-CIC-IDS2018"])]
    report["falls_back_v1"] = report["episodes_v1_sample"] < 3
    report["falls_back_full"] = report["episodes_full_data"] < 3
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report.sort_values(["source", "gap_min", "attack"]).to_csv(REPORT, index=False)
    log(f"Report: {REPORT}")
    print(report.pivot_table(index=["source", "attack"], columns="gap_min",
                             values=["episodes_v1_sample", "episodes_full_data"]).to_string())


if __name__ == "__main__":
    main()

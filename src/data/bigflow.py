"""
BigFlow source-aware data layer.

BigFlow-NIDS-V2 merges four public NetFlow captures recorded years apart.
Each merged_part_*.parquet file mixes rows from several of them, so the
origin of a flow has to be recovered from its timestamp:

    UNSW-NB15        2015            ~5% attacks
    CSE-CIC-IDS2018  Feb-Apr 2018    ~12% attacks
    BoT-IoT          May-Jun 2018    ~100% attacks
    ToN-IoT          2019            ~39% attacks

Mixing sources creates a shortcut: anything that identifies the source
(timestamp, TTL, TCP window, IP range) also predicts the label. Experiments
should therefore run per source, and report source identification
separately.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


SOURCE_PERIODS = [
    # (source, start inclusive, end exclusive) - UTC
    ("UNSW-NB15", "2000-01-01", "2016-01-01"),
    ("CSE-CIC-IDS2018", "2018-01-01", "2018-04-15"),
    ("BoT-IoT", "2018-04-15", "2019-01-01"),
    ("ToN-IoT", "2019-01-01", "2030-01-01"),
]

# Columns that identify a flow, its time or its origin rather than its
# behaviour. They must never be model inputs.
IDENTIFIER_COLUMNS = [
    "flow_start_milliseconds",
    "flow_end_milliseconds",
    "ipv4_src_addr",
    "ipv4_dst_addr",
    "l4_src_port",
    "dns_query_id",
]

TARGET_COLUMNS = ["label", "attack"]


def infer_source(timestamps_ms):
    """Map flow start timestamps (ms) to their source dataset name."""
    timestamps_ms = np.asarray(timestamps_ms)
    source = np.full(len(timestamps_ms), "unknown", dtype=object)

    for name, start, end in SOURCE_PERIODS:
        lo = pd.Timestamp(start).value // 1_000_000
        hi = pd.Timestamp(end).value // 1_000_000
        source[(timestamps_ms >= lo) & (timestamps_ms < hi)] = name

    return source


def bigflow_dir():
    """BigFlow directory from configs/config.yaml (datasets.bigflow.path)."""
    path = Path(Config().get_dataset_path("bigflow"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def list_files(directory):
    return sorted(
        glob.glob(str(Path(directory) / "*.parquet")),
        key=lambda path: int(Path(path).stem.split("_")[-1]),
    )


def read_file(path, columns=None):
    """Read one BigFlow file with lower-case column names and a source column."""
    table = pq.read_table(path, columns=columns)
    df = table.to_pandas()
    df.columns = df.columns.str.lower()
    df["source"] = infer_source(df["flow_start_milliseconds"].to_numpy())
    return df


def behaviour_columns(columns):
    """Every column except identifiers, targets and the source name."""
    excluded = set(IDENTIFIER_COLUMNS) | set(TARGET_COLUMNS) | {"source"}
    return [col for col in columns if col not in excluded]


def build_working_sample(
    directory,
    attack_cap=20_000,
    benign_cap=100_000,
    random_state=42,
    extra_columns=None,
):
    """
    Stratified sample over all files, capped per (source, attack) class.

    Returns the sample with ``file_id`` / ``file_row`` columns so rows can be
    joined back to features computed on the complete data.
    """
    files = list_files(directory)
    rng = np.random.default_rng(random_state)

    # First pass: class counts, to turn caps into per-file sampling rates.
    counts = []
    for path in files:
        meta = read_file(path, ["FLOW_START_MILLISECONDS", "Attack"])
        counts.append(meta.groupby(["source", "attack"]).size())
    totals = pd.concat(counts).groupby(level=[0, 1]).sum()

    caps = pd.Series(
        np.where(totals.index.get_level_values(1) == "Benign", benign_cap, attack_cap),
        index=totals.index,
    )
    rate = (caps / totals).clip(upper=1.0)

    parts = []
    for file_id, path in enumerate(files):
        df = read_file(path)
        df["file_id"] = file_id
        df["file_row"] = np.arange(len(df))
        keep = rng.random(len(df)) < rate.reindex(
            pd.MultiIndex.from_arrays([df["source"], df["attack"]])
        ).to_numpy()
        parts.append(df.loc[keep])

    sample = pd.concat(parts, ignore_index=True)
    return sample.sort_values("flow_start_milliseconds", ignore_index=True)

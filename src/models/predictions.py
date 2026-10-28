"""
Per-flow validation and test predictions, saved for every run and seed.

The secondary response-time metrics (PROTOCOL_FREEZE.md, section 6: time to detect, episode
detection rate, benign alerts per hour) are computed from these files after the run. Validation
rows are kept so that alert thresholds can be recomputed exactly (pooled (1 - FPR) quantile of
validation benign scores).

One file per (experiment, dataset, source, seed, protocol[, variant]); one float32 score column
per model/representation. Files are never overwritten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROW_COLUMNS = ["row", "split", "t_ms", "t_end_ms", "episode", "episode_start_ms", "mode", "family", "label",
               "design_weight", "in_cross", "in_fallback"]


def episode_start_ms(episode, bounds):
    """Full-data start of each row's episode ('key|n' ids from map_episodes); -1 where unknown."""
    out = np.full(len(episode), -1, dtype=np.int64)
    if bounds is None:
        return out
    starts = dict(zip(bounds["key"] + "|" + bounds["episode"].astype(str), bounds["start_ms"]))
    ep = pd.Series(episode, dtype=object)
    mapped = ep.map(starts)
    known = mapped.notna().to_numpy()
    out[known] = mapped[known].astype(np.int64).to_numpy()
    return out


def prediction_frame(sdf, split, episode, mode, subsets, scores, weights, bounds=None):
    """
    sdf: the source frame (rows in the order used for training); split/episode/mode: arrays over
    its rows; subsets: masks from test_subsets(); scores: {column name: score array over all rows};
    weights: design (or prevalence) weight per row. Returns validation and test rows only.
    """
    keep = np.flatnonzero(np.isin(split, ["validation", "test"]))
    episode = np.asarray(episode, dtype=object)
    t = (sdf["flow_start_milliseconds"].to_numpy().astype(np.int64) if "flow_start_milliseconds" in sdf
         else np.full(len(sdf), -1, dtype=np.int64))
    t_end = (sdf["flow_end_milliseconds"].to_numpy().astype(np.int64) if "flow_end_milliseconds" in sdf
             else np.full(len(sdf), -1, dtype=np.int64))
    frame = pd.DataFrame({
        "row": keep.astype(np.int32),
        "split": pd.Categorical(np.asarray(split, dtype=object)[keep]),
        "t_ms": t[keep],
        "t_end_ms": t_end[keep],
        "episode": pd.Categorical(pd.Series(episode[keep], dtype=object).fillna("").astype(str)),
        "episode_start_ms": episode_start_ms(episode[keep], bounds),
        "mode": pd.Categorical(np.asarray(mode, dtype=object)[keep].astype(str)),
        "family": pd.Categorical(sdf["attack"].to_numpy()[keep]),
        "label": sdf["label"].to_numpy()[keep].astype(np.int8),
        "design_weight": np.asarray(weights, dtype=np.float32)[keep],
        # random-row splits have only the "all" subset: no row is in (i) or (ii)
        "in_cross": np.asarray(subsets.get("cross-episode", np.zeros(len(sdf), bool)))[keep],
        "in_fallback": np.asarray(subsets.get("fallback", np.zeros(len(sdf), bool)))[keep],
    })
    for name, s in scores.items():
        frame[name] = np.asarray(s, dtype=np.float32)[keep]
    return frame


def write_predictions(path: Path, frame: pd.DataFrame):
    """Write once; refuse to overwrite an existing prediction file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"prediction file exists, not overwriting: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.rename(path)
    return path


def slug(text):
    return "".join(c if c.isalnum() else "_" for c in str(text)).strip("_").replace("__", "_")

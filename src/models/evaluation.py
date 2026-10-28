"""Leakage-aware splitting, threshold tuning and detection metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


def blocked_time_split(
    timestamps,
    strata,
    block_ms=600_000,
    validation_fraction=0.15,
    test_fraction=0.15,
    random_state=42,
    max_tries=200,
):
    """
    Split rows into train / validation / test by whole time blocks.

    Flows from the same burst are near-duplicates. A random row split puts
    siblings of every test flow into training and inflates scores. Assigning
    whole ``block_ms`` windows to one split removes that leakage.

    Blocks are assigned to splits uniformly at random (with probabilities
    1 - val - test, val, test). Any stratum (e.g. attack family) with at least
    three blocks that is missing from a split then gets one of its blocks moved
    there, so every such stratum appears in train, validation and test.

    Note: an earlier version used ``StratifiedGroupKFold``, whose shuffling
    barely changed the assignment between seeds (92% test-set overlap) and
    put validation benign traffic in an unrepresentative period.

    Returns an array of "train" / "validation" / "test".
    """
    timestamps = np.asarray(timestamps)
    strata = np.asarray(strata)
    blocks = (timestamps // block_ms).astype(np.int64)
    unique_blocks, block_index = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(random_state)

    # Which blocks contain each stratum.
    pairs = pd.DataFrame({"block": block_index, "stratum": strata}).drop_duplicates()
    blocks_of = pairs.groupby("stratum")["block"].apply(np.asarray)
    required = [s for s, b in blocks_of.items() if len(b) >= 3]

    probabilities = [1 - validation_fraction - test_fraction, validation_fraction, test_fraction]
    block_split = rng.choice(3, size=len(unique_blocks), p=probabilities)

    # Repair: a stratum missing from a split gets one of its blocks moved there,
    # taken from the split where that stratum has the most blocks.
    for _ in range(max_tries):
        changed = False
        for stratum in rng.permutation(required):
            own = blocks_of[stratum]
            counts = np.bincount(block_split[own], minlength=3)
            for missing in np.flatnonzero(counts == 0):
                donor = int(np.argmax(counts))
                candidates = own[block_split[own] == donor]
                block_split[rng.choice(candidates)] = missing
                counts = np.bincount(block_split[own], minlength=3)
                changed = True
        if not changed:
            labels = np.array(["train", "validation", "test"], dtype=object)
            return labels[block_split[block_index]]

    raise RuntimeError("could not place every stratum in all three splits; use larger blocks")


def tune_threshold(y_true, scores, metric="f1", sample_weight=None):
    """
    Pick the decision threshold maximising F1 on validation data.

    Pass ``sample_weight`` (e.g. prevalence weights) when the validation
    rows do not have the real attack prevalence; otherwise the threshold is
    calibrated for the wrong class mix.
    """
    if metric != "f1":
        raise ValueError("Only metric='f1' is supported")

    precision, recall, thresholds = precision_recall_curve(
        y_true, scores, sample_weight=sample_weight
    )
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    # precision_recall_curve returns one more P/R point than thresholds.
    best = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]) if len(thresholds) else 0.5


def evaluate_binary(y_true, scores, threshold=0.5, families=None, name=None):
    """Binary detection metrics plus per-family recall (detection rate)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": (tp + tn) / len(y_true),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores),
        "FPR": fp / max(fp + tn, 1),
        "FNR": fn / max(fn + tp, 1),
    }

    if families is not None:
        families = np.asarray(families)
        for family in sorted(pd.unique(families[y_true == 1])):
            mask = families == family
            metrics[f"Recall_{family}"] = y_pred[mask].mean()

    return metrics


def assign_episodes(timestamps, keys, gap_ms=300_000):
    """
    Number attack episodes: within each key (e.g. source + attack family),
    a new episode starts when consecutive flows are more than ``gap_ms`` apart.

    Returns an array of episode ids such as "ToN-IoT|scanning|3".
    """
    frame = pd.DataFrame({"t": np.asarray(timestamps), "key": np.asarray(keys)})
    frame = frame.sort_values(["key", "t"], kind="stable")

    new_episode = (frame["key"] != frame["key"].shift()) | (
        frame["t"].diff() > gap_ms
    )
    number = new_episode.groupby(frame["key"]).cumsum()

    episode = frame["key"].astype(str) + "|" + number.astype(str)
    return episode.reindex(range(len(frame))).to_numpy()


TRAIN_MIN_SHARE = 0.50
HELD_OUT_SHARE_RANGE = (0.10, 0.25)


def rows_stratum(stratum, rows):
    return str(stratum.iloc[rows[0]]) if hasattr(stratum, "iloc") else str(stratum[rows[0]])


def assign_episodes_constrained(sizes, rng, validation_fraction=0.15, test_fraction=0.15,
                                max_tries=1000):
    """
    Assign whole episodes (``sizes``: Series episode -> flows) to splits.

    Constrained random assignment: each episode independently goes to train /
    validation / test with probabilities (1 - val - test, val, test); a draw is
    accepted if every split gets at least one episode, training holds at least
    50% of the flows, and validation and test each hold 10-25%. Up to
    ``max_tries`` draws. If none is accepted, largest-first fallback: the largest
    episode goes to training and every further episode (in decreasing size) to the
    split furthest below its target share, keeping one episode for each split.

    Returns (assignment dict, method 'random' | 'largest-first', draws used).
    """
    names = sizes.index.to_numpy()
    counts = sizes.to_numpy(dtype=float)
    total = counts.sum()
    labels = np.array(["train", "validation", "test"], dtype=object)
    p = [1 - validation_fraction - test_fraction, validation_fraction, test_fraction]
    lo, hi = HELD_OUT_SHARE_RANGE
    for draw in range(1, max_tries + 1):
        choice = rng.choice(3, size=len(names), p=p)
        share = np.bincount(choice, weights=counts, minlength=3) / total
        if (np.bincount(choice, minlength=3) > 0).all() and share[0] >= TRAIN_MIN_SHARE \
                and lo <= share[1] <= hi and lo <= share[2] <= hi:
            return dict(zip(names, labels[choice])), "random", draw

    order = np.argsort(-counts, kind="stable")
    target = np.array(p)
    filled = np.zeros(3)
    assigned = {}
    for rank, i in enumerate(order):
        remaining = len(order) - rank
        empty = [k for k in (1, 2) if filled[k] == 0 and not any(
            v == labels[k] for v in assigned.values())]
        if rank == 0:
            k = 0
        elif remaining <= len(empty):
            k = empty[0]
        else:
            k = int(np.argmax(target - filled / total))
        assigned[names[i]] = labels[k]
        filled[k] += counts[i]
    return assigned, "largest-first", max_tries


def episode_split(
    timestamps,
    source,
    attack,
    gap_ms=300_000,
    benign_block_ms=1_800_000,
    validation_fraction=0.15,
    test_fraction=0.15,
    min_episodes=3,
    random_state=42,
    episodes=None,
    assignment_log=None,
):
    """
    Split so that every attack episode lies entirely in one split.

    Episodes of a stratum are assigned by ``assign_episodes_constrained``
    (constrained random assignment with a largest-first fallback). Pass a list as
    ``assignment_log`` to receive one record per stratum (method used, draws).

    ``episodes`` (optional): precomputed episode ids for every row (attack rows;
    benign entries are ignored), e.g. from ``map_episodes`` on full-data episode
    boundaries. When omitted, episodes are built from the given timestamps.

    * Attack flows are grouped into episodes (gap > ``gap_ms`` starts a new one).
    * Benign flows are grouped into ``benign_block_ms`` time blocks per source.
    * Within each (source, attack) stratum, whole episodes are assigned at random
      under constraints (see ``assign_episodes_constrained``).
    * Strata with fewer than ``min_episodes`` episodes cannot be split by
      episode. Their flows are split chronologically instead (earliest to
      train, latest to test) and flagged with mode "chronological".

    Returns (split, episode_id, mode) arrays.
    """
    timestamps = np.asarray(timestamps)
    source = np.asarray(source).astype(str)
    attack = np.asarray(attack).astype(str)
    rng = np.random.default_rng(random_state)

    stratum = pd.Series(source) + "|" + pd.Series(attack)
    is_benign = attack == "Benign"

    if episodes is None:
        episode = assign_episodes(timestamps, stratum.to_numpy(), gap_ms)
    else:
        episode = np.asarray(episodes, dtype=object)
    benign_block = stratum + "|block" + pd.Series(timestamps // benign_block_ms).astype(str)
    episode = np.where(is_benign, benign_block.to_numpy(), episode)

    split = np.empty(len(timestamps), dtype=object)
    mode = np.full(len(timestamps), "episode", dtype=object)

    for _, rows in pd.Series(np.arange(len(timestamps))).groupby(stratum.to_numpy()):
        rows = rows.to_numpy()
        episodes = pd.Series(episode[rows]).value_counts()

        if len(episodes) < min_episodes:
            order = rows[np.argsort(timestamps[rows], kind="stable")]
            n = len(order)
            n_test = int(round(n * test_fraction))
            n_val = int(round(n * validation_fraction))
            split[order[: n - n_test - n_val]] = "train"
            split[order[n - n_test - n_val : n - n_test]] = "validation"
            split[order[n - n_test :]] = "test"
            mode[rows] = "chronological"
            continue

        assignment, method, tries = assign_episodes_constrained(
            episodes, rng, validation_fraction, test_fraction)
        if assignment_log is not None:
            assignment_log.append({"stratum": rows_stratum(stratum, rows), "episodes": len(episodes),
                                   "method": method, "tries": tries})
        split[rows] = [assignment[e] for e in episode[rows]]

    return split, episode, mode


def prevalence_weights(source, attack, true_counts):
    """
    Weights that restore each (source, attack) class to its real frequency.

    ``true_counts`` maps (source, attack) -> number of flows in the complete
    data. Sampling caps per class distort the attack share (e.g. ToN-IoT is
    38% attacks in reality but ~65% in a capped sample), which inflates
    precision and F1.
    """
    keys = pd.Series(list(zip(np.asarray(source), np.asarray(attack))))
    sample_counts = keys.value_counts()
    true = keys.map(true_counts).astype(float)
    return (true / keys.map(sample_counts)).to_numpy()


def evaluate_detection(
    y_true,
    scores,
    val_benign_scores,
    families=None,
    weights=None,
    target_fprs=(0.01, 0.001),
    f1_threshold=None,
    name=None,
):
    """
    Operational detection metrics.

    * PR-AUC / ROC-AUC / F1 use ``weights`` so the test set has the real
      attack prevalence.
    * For each target false-positive rate, the alert threshold is the
      matching quantile of VALIDATION benign scores (as a SOC would set it);
      the test FPR actually achieved and the detection rate (overall and per
      family) are reported.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    weights = np.ones(len(y_true)) if weights is None else np.asarray(weights, float)

    metrics = {
        "Model": name,
        "PR_AUC_w": average_precision_score(y_true, scores, sample_weight=weights),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "attack_share_w": float(np.average(y_true, weights=weights)),
    }

    if f1_threshold is not None:
        y_pred = (scores >= f1_threshold).astype(int)
        metrics["F1_w"] = f1_score(y_true, y_pred, sample_weight=weights, zero_division=0)
        metrics["Precision_w"] = precision_score(
            y_true, y_pred, sample_weight=weights, zero_division=0)

    benign = y_true == 0
    for fpr in target_fprs:
        tag = f"{fpr * 100:g}pct"
        threshold = np.quantile(val_benign_scores, 1 - fpr)
        y_pred = scores > threshold
        metrics[f"TPR@FPR{tag}"] = float(np.average(y_pred[~benign], weights=weights[~benign]))
        metrics[f"testFPR@{tag}"] = float(y_pred[benign].mean())

        if families is not None:
            families = np.asarray(families)
            for family in sorted(pd.unique(families[~benign])):
                mask = families == family
                metrics[f"TPR@FPR{tag}_{family}"] = float(y_pred[mask].mean())

    return metrics


def episode_bounds(timestamps, gap_ms):
    """
    Episodes of one class on the full data: a new episode starts after a gap of
    more than ``gap_ms``. Returns (start_ms, end_ms, flows) arrays.
    """
    t = np.sort(np.asarray(timestamps, dtype=np.int64))
    if len(t) == 0:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.int64)
    breaks = np.flatnonzero(np.diff(t) > gap_ms)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(t) - 1]
    return t[starts], t[ends], ends - starts + 1


def map_episodes(timestamps, keys, bounds):
    """
    Map rows onto full-data episodes.

    bounds: DataFrame with columns key, episode, start_ms, end_ms (one row per
    episode). Rows whose key has no bounds (e.g. benign) get None. A timestamp
    taken from the full data always lies inside one episode of its key.
    """
    timestamps = np.asarray(timestamps, dtype=np.int64)
    keys = np.asarray(keys).astype(str)
    out = np.full(len(keys), None, dtype=object)
    for key, b in bounds.groupby("key"):
        rows = np.flatnonzero(keys == key)
        if len(rows) == 0:
            continue
        b = b.sort_values("start_ms")
        idx = np.searchsorted(b["start_ms"].to_numpy(), timestamps[rows], side="right") - 1
        idx = np.clip(idx, 0, len(b) - 1)
        inside = timestamps[rows] <= b["end_ms"].to_numpy()[idx]
        if not inside.all():
            raise ValueError(f"{(~inside).sum()} rows of {key} fall outside every episode")
        out[rows] = [f"{key}|{e}" for e in b["episode"].to_numpy()[idx]]
    return out


def split_validation(split, episode, mode, strata, timestamps, random_state=0):
    """
    Divide the validation rows into two disjoint halves:

    * "validation_select"    - used to choose experts / calibrate a detector;
    * "validation_threshold" - used only to set the alert thresholds.

    Within each stratum (e.g. source|class), whole validation episodes are
    shuffled and each is assigned to the half that currently holds fewer flows,
    so no episode is split across halves. Strata with a single validation
    episode, or in the chronological fallback, are split chronologically: the
    earlier half of their validation rows selects, the later half sets thresholds.
    Rows outside validation keep their split label.
    """
    split = np.asarray(split, dtype=object).copy()
    episode = np.asarray(episode, dtype=object)
    mode = np.asarray(mode, dtype=object)
    strata = np.asarray(strata).astype(str)
    timestamps = np.asarray(timestamps)
    rng = np.random.default_rng(random_state)
    val = np.flatnonzero(split == "validation")
    out = split.copy()

    for stratum in np.unique(strata[val]):
        rows = val[strata[val] == stratum]
        names = pd.unique(episode[rows])
        if len(names) < 2 or (mode[rows] == "chronological").any():
            order = rows[np.argsort(timestamps[rows], kind="stable")]
            half = len(order) // 2
            out[order[:half]] = "validation_select"
            out[order[half:]] = "validation_threshold"
            if half == 0:  # a single row: keep it for thresholds
                out[order] = "validation_threshold"
            continue
        sizes = pd.Series(episode[rows]).value_counts()
        names = names.copy()
        rng.shuffle(names)
        count = {"validation_select": 0, "validation_threshold": 0}
        assignment = {}
        for i, name in enumerate(names):
            # the first two episodes seed both halves
            target = (["validation_select", "validation_threshold"][i] if i < 2
                      else min(count, key=count.get))
            assignment[name] = target
            count[target] += sizes[name]
        out[rows] = [assignment[e] for e in episode[rows]]
    return out


def _near_any(t, starts, ends, margin):
    """True where t lies within ``margin`` of any interval [start, end]."""
    t = np.asarray(t, dtype=np.int64)
    if len(starts) == 0 or len(t) == 0:
        return np.zeros(len(t), bool)
    lo = np.asarray(starts, np.int64) - margin
    hi = np.asarray(ends, np.int64) + margin
    order = np.argsort(lo)
    lo, hi = lo[order], hi[order]
    # merge overlapping intervals
    merged_lo, merged_hi = [lo[0]], [hi[0]]
    for a, b in zip(lo[1:], hi[1:]):
        if a <= merged_hi[-1]:
            merged_hi[-1] = max(merged_hi[-1], b)
        else:
            merged_lo.append(a)
            merged_hi.append(b)
    merged_lo, merged_hi = np.array(merged_lo), np.array(merged_hi)
    idx = np.searchsorted(merged_lo, t, side="right") - 1
    ok = idx >= 0
    out = np.zeros(len(t), bool)
    out[ok] = t[ok] <= merged_hi[idx[ok]]
    return out


def unit_ranges(split, episode, mode, strata, timestamps, episode_ranges=None):
    """
    Time range of every assignment unit: an episode (full-data range from
    ``episode_ranges`` when given, else the range of its rows), or, for strata in
    the chronological fallback, each contiguous split segment of the stratum.
    Returns a DataFrame: stratum, unit, split, start, end.
    """
    frame = pd.DataFrame({"split": split, "episode": episode, "mode": mode,
                          "stratum": strata, "t": np.asarray(timestamps, np.int64)})
    frame["unit"] = np.where(frame["mode"] == "chronological",
                             frame["stratum"] + "|chron|" + frame["split"].astype(str),
                             frame["episode"].astype(str))
    g = frame.groupby(["stratum", "unit", "split"], sort=False)["t"].agg(["min", "max"]).reset_index()
    g = g.rename(columns={"min": "start", "max": "end"})
    if episode_ranges is not None:
        known = g["unit"].isin(episode_ranges.keys())
        g.loc[known, "start"] = [episode_ranges[u][0] for u in g.loc[known, "unit"]]
        g.loc[known, "end"] = [episode_ranges[u][1] for u in g.loc[known, "unit"]]
    return g


def apply_embargo(split, episode, mode, strata, timestamps, embargo_ms, episode_ranges=None,
                  source=None, benign_label="Benign", fallback_val_test_ms=60_000,
                  src_ip=None, dst_ip=None, benign_host_window_ms=60_000):
    """
    Purge flows close to a unit of the same class assigned to another split.

    Attack classes (and benign blocks among themselves):
    * test flows are never purged;
    * a training flow of class c is purged if it lies within ``embargo_ms`` of any
      validation or test unit of class c;
    * a validation flow of class c is purged if it lies within ``embargo_ms`` of any
      test unit of class c - except for classes in the chronological fallback, where
      the validation/test buffer is ``fallback_val_test_ms`` (default 1 min, the
      longest host-feature window), so short single-episode classes keep validation data.

    Benign flows near attacks (when ``src_ip``/``dst_ip`` are given): a benign
    training flow is purged if its source or destination IP appears (as source or
    destination) in a held-out (validation or test) attack flow of the same source
    dataset that starts at most ``benign_host_window_ms`` before it.

    Purged rows get the split label "purged". Returns the new split array.
    """
    split = np.asarray(split, dtype=object).copy()
    strata = np.asarray(strata).astype(str)
    t = np.asarray(timestamps, np.int64)
    if embargo_ms <= 0:
        return split
    mode = np.asarray(mode, dtype=object)
    units = unit_ranges(split, episode, mode, strata, t, episode_ranges)
    source = (np.asarray(source).astype(str) if source is not None
              else np.array([x.split("|")[0] for x in strata]))
    purge = np.zeros(len(split), bool)

    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        u = units[units["stratum"] == stratum]
        held_u = u[u["split"].isin(["validation", "test"])]
        test_u = u[u["split"] == "test"]
        chronological = (mode[rows] == "chronological").all()
        val_margin = fallback_val_test_ms if chronological else embargo_ms
        train_rows = rows[split[rows] == "train"]
        val_rows = rows[split[rows] == "validation"]
        purge[train_rows] |= _near_any(t[train_rows], held_u["start"], held_u["end"], embargo_ms)
        purge[val_rows] |= _near_any(t[val_rows], test_u["start"], test_u["end"], val_margin)

    if src_ip is not None and dst_ip is not None:
        src_ip = np.asarray(src_ip).astype(str)
        dst_ip = np.asarray(dst_ip).astype(str)
        benign = np.array([x.endswith("|" + benign_label) for x in strata])
        held_attack = ~benign & np.isin(split, ["validation", "test"])
        for src in np.unique(source):
            in_src = source == src
            ha = np.flatnonzero(held_attack & in_src)
            if len(ha) == 0:
                continue
            times_by_ip = (pd.DataFrame({"ip": np.r_[src_ip[ha], dst_ip[ha]], "t": np.r_[t[ha], t[ha]]})
                           .groupby("ip")["t"].apply(lambda x: np.sort(x.to_numpy())).to_dict())
            for i in np.flatnonzero(benign & in_src & (split == "train")):
                for ip in (src_ip[i], dst_ip[i]):
                    times = times_by_ip.get(ip)
                    if times is None:
                        continue
                    k = np.searchsorted(times, t[i], side="right")
                    if k > 0 and t[i] - times[k - 1] <= benign_host_window_ms:
                        purge[i] = True
                        break

    split[purge] = "purged"
    return split


def chronological_split(timestamps, strata, validation_fraction=0.15, test_fraction=0.15):
    """Reference protocol: every stratum split by time (earliest 70% train, latest 15% test)."""
    t = np.asarray(timestamps)
    strata = np.asarray(strata).astype(str)
    split = np.empty(len(t), dtype=object)
    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        order = rows[np.argsort(t[rows], kind="stable")]
        n = len(order)
        n_test, n_val = int(round(n * test_fraction)), int(round(n * validation_fraction))
        split[order[: n - n_test - n_val]] = "train"
        split[order[n - n_test - n_val: n - n_test]] = "validation"
        split[order[n - n_test:]] = "test"
    return split

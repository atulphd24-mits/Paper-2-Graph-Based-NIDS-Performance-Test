"""
Statistics protocol for paired, seed-matched comparisons.

For each comparison (candidate vs baseline, paired by seed):
* two-sided Wilcoxon signed-rank test on the per-seed differences;
* paired bootstrap 95% confidence interval of the mean difference
  (resampling seeds with replacement, 10,000 resamples, percentile method);
* Holm-Bonferroni adjustment of the p-values within each explicitly defined
  group of related tests (one family of hypotheses).

A group is defined by the caller through ``group_keys``: all comparisons that
share the same values of ``group_keys`` (and the same metric) form one family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (NaNs are left out and stay NaN)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if len(ok) == 0:
        return adjusted
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_ci(differences, n_resamples=10_000, level=0.95, seed=0):
    """Percentile bootstrap CI of the mean of paired differences."""
    d = np.asarray(differences, dtype=float)
    if len(d) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_resamples, len(d)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def wilcoxon_p(differences):
    d = np.asarray(differences, dtype=float)
    if len(d) < 5 or not np.any(d != 0):
        return np.nan
    return float(wilcoxon(d).pvalue)


def compare(frame, condition_col, baseline, metrics, group_keys, seed_col="seed",
            candidates=None, n_resamples=10_000):
    """
    Paired comparisons of every condition against ``baseline``.

    frame: one row per (group..., condition, seed) with metric columns.
    condition_col: column holding the condition name (e.g. representation, model).
    group_keys: columns defining one family of tests (e.g. ["source", "model"]).
    Returns one row per (group, candidate, metric) with the mean difference, its
    bootstrap CI, the raw Wilcoxon p, the Holm-adjusted p within the family,
    the family size and the number of seeds where the candidate is better.
    """
    rows = []
    keys = list(group_keys)
    for group, g in frame.groupby(keys, sort=False) if keys else [((), frame)]:
        group = group if isinstance(group, tuple) else (group,)
        base = g[g[condition_col] == baseline].set_index(seed_col)
        names = candidates or [c for c in g[condition_col].unique() if c != baseline]
        for metric in metrics:
            family = []
            for name in names:
                cand = g[g[condition_col] == name].set_index(seed_col)
                seeds = base.index.intersection(cand.index)
                if len(seeds) == 0:
                    continue
                d = (cand.loc[seeds, metric] - base.loc[seeds, metric]).to_numpy(float)
                lo, hi = bootstrap_ci(d, n_resamples)
                family.append({
                    **dict(zip(keys, group)), "metric": metric, "candidate": name,
                    "baseline": baseline, "seeds": len(d),
                    "baseline_mean": float(base.loc[seeds, metric].mean()),
                    "candidate_mean": float(cand.loc[seeds, metric].mean()),
                    "mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi,
                    "wins": int((d > 0).sum()), "p_raw": wilcoxon_p(d),
                })
            adj = holm([r["p_raw"] for r in family])
            for r, a in zip(family, adj):
                r["p_holm"] = a
                r["family_size"] = len(family)
            rows += family
    return pd.DataFrame(rows)


def episode_bootstrap_tpr(detected, weights, episodes, n_resamples=2000, level=0.95, seed=0):
    """
    Cluster bootstrap of a (weighted) detection rate over attack episodes.

    detected: bool per attack flow (score above the alert threshold);
    weights: prevalence weight per flow; episodes: episode id per flow.
    Episodes are resampled with replacement (flows are never resampled on their
    own), so the interval reflects how many independent attack instances there are.
    Returns (tpr, ci_low, ci_high, n_episodes).
    """
    detected = np.asarray(detected, float)
    weights = np.asarray(weights, float)
    frame = pd.DataFrame({"e": np.asarray(episodes, dtype=object).astype(str),
                          "hit": detected * weights, "w": weights})
    per = frame.groupby("e")[["hit", "w"]].sum()
    n = len(per)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    tpr = per["hit"].sum() / per["w"].sum()
    if n < 2:
        return float(tpr), np.nan, np.nan, n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    hit, w = per["hit"].to_numpy()[idx].sum(1), per["w"].to_numpy()[idx].sum(1)
    boot = hit / w
    a = (1 - level) / 2
    return float(tpr), float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a)), n

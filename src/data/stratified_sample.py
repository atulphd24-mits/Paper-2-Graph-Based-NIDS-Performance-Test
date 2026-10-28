"""
Episode-stratified capped sampling with design weights.

For one attack class with episodes e = 1..m of sizes s_e (full data) and a cap C:
  * if the class has at most C flows, keep all of them (weight 1);
  * otherwise keep k_e = min(k_min, s_e) flows chosen at random from every
    episode, then fill the remaining quota R = C - sum(k_e) at random from the
    flows not yet chosen (pooled over episodes). If sum(k_e) > C, the mandatory
    part is itself capped by keeping one flow per episode first (see code).

Every flow of episode e then has the same inclusion probability
    pi_e = k_e / s_e + (1 - k_e / s_e) * R / N_rest,    N_rest = sum(s_e - k_e)
and gets the design weight 1 / pi_e. Weighted sums over the sample are unbiased
for the corresponding full-data sums, so prevalence-weighted metrics estimate the
population rates even though small episodes are over-represented in the sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_episode_sample(episode, cap, k_min=20, rng=None):
    """
    episode: array of episode ids, one per flow of a single class (full data).
    Returns (selected positions, design weight per selected position).
    """
    rng = rng or np.random.default_rng(0)
    episode = np.asarray(episode)
    n = len(episode)
    if n <= cap:
        return np.arange(n), np.ones(n)

    codes, uniques = pd.factorize(episode)
    sizes = np.bincount(codes)
    k = np.minimum(k_min, sizes)
    if k.sum() > cap:
        # Too many episodes for k_min each: give every episode the same reduced share.
        k = np.minimum(sizes, max(1, cap // len(sizes)))
    order = rng.permutation(n)
    rank_in_episode = pd.Series(codes[order]).groupby(codes[order]).cumcount().to_numpy()
    mandatory = order[rank_in_episode < k[codes[order]]]

    rest_mask = np.ones(n, bool)
    rest_mask[mandatory] = False
    rest = np.flatnonzero(rest_mask)
    remaining = max(0, cap - len(mandatory))
    fill = rng.choice(rest, size=min(remaining, len(rest)), replace=False) if remaining else np.array([], int)

    selected = np.sort(np.r_[mandatory, fill]).astype(int)
    n_rest = len(rest)
    fill_rate = (len(fill) / n_rest) if n_rest else 0.0
    pi = k / sizes + (1 - k / sizes) * fill_rate
    return selected, 1.0 / pi[codes[selected]]

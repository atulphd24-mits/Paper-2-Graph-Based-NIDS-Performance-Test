"""
Neighbourhood Feature Extractor

Author : Atul Chauhan
Project : SOC Threat Detection Framework

Structural centralities (degree, PageRank, ...) are nearly identical for
every node of a k-NN graph, so they cannot separate benign from attack
flows. This module instead describes *who a flow's neighbours are*:

1. Label-aware features (training labels only, leakage-safe)
   - share of labelled neighbours that are attacks (plain and
     distance-weighted)
   - share of labelled neighbours in each attack family
   Neighbours are looked up among training nodes only. Training nodes
   exclude themselves (leave-one-out), so train and test rows see the
   same kind of evidence.

2. Label-free context features (all nodes)
   - mean of neighbours' features for selected columns
   - node value minus neighbourhood mean (local deviation)
   - distance to the 1st / k-th neighbour and mean k-NN distance
   - local outlier factor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


class NeighborhoodFeatureExtractor:

    def __init__(self, k=10, metric="euclidean", n_jobs=-1, inductive=False):
        """
        ``inductive=True`` builds the label-free context features (k-NN distances,
        LOF, neighbour statistics) from training flows only: validation and test
        flows are queried against the training index and never influence the
        features of any other flow. The default (False) reproduces the earlier,
        transductive behaviour, where the index and LOF used all rows.
        """
        if k < 1:
            raise ValueError("k must be at least 1")

        self.k = k
        self.metric = metric
        self.n_jobs = n_jobs
        self.inductive = inductive

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _knn(self, reference, query, exclude_self_rows=None):
        """
        k-NN of ``query`` rows among ``reference`` rows.

        ``exclude_self_rows[i]`` is the reference row equal to query row i
        (or -1), which is removed from its own neighbour list.
        """
        n_neighbors = min(self.k + 1, len(reference))
        index = NearestNeighbors(
            n_neighbors=n_neighbors, metric=self.metric, n_jobs=self.n_jobs
        ).fit(reference)
        distances, indices = index.kneighbors(query)

        if exclude_self_rows is None:
            return distances[:, : self.k], indices[:, : self.k]

        is_self = indices == np.asarray(exclude_self_rows)[:, None]
        order = np.argsort(is_self, axis=1, kind="stable")[:, : n_neighbors - 1]
        order = order[:, : self.k]

        return (
            np.take_along_axis(distances, order, axis=1),
            np.take_along_axis(indices, order, axis=1),
        )

    # --------------------------------------------------------
    # Label-aware features
    # --------------------------------------------------------

    def label_features(self, X, labels, train_mask, families=None):
        """
        Neighbour label statistics using training labels only.

        Parameters
        ----------
        X : array (n, d)          transformed feature matrix, all nodes
        labels : array (n,)       binary labels (1 = attack); only
                                  ``labels[train_mask]`` is read
        train_mask : bool (n,)
        families : array (n,)     optional attack-family names; only
                                  training rows are read
        """
        X = np.asarray(X)
        train_mask = np.asarray(train_mask, dtype=bool)
        train_rows = np.flatnonzero(train_mask)

        # Position of each node inside the training reference set (or -1).
        position = np.full(len(X), -1)
        position[train_rows] = np.arange(len(train_rows))

        distances, neighbours = self._knn(
            X[train_rows], X, exclude_self_rows=position
        )

        train_labels = np.asarray(labels)[train_rows].astype(float)
        neighbour_labels = train_labels[neighbours]

        weights = 1.0 / (distances + 1e-6)
        out = {
            "nbr_attack_ratio": neighbour_labels.mean(axis=1),
            "nbr_attack_ratio_weighted": (
                (neighbour_labels * weights).sum(axis=1) / weights.sum(axis=1)
            ),
            "nbr_nearest_is_attack": neighbour_labels[:, 0],
        }

        if families is not None:
            train_families = np.asarray(families)[train_rows]
            neighbour_families = train_families[neighbours]
            for family in sorted(pd.unique(train_families)):
                out[f"nbr_family_ratio_{family}"] = (
                    neighbour_families == family
                ).mean(axis=1)

        return pd.DataFrame(out)

    # --------------------------------------------------------
    # Label-free context features
    # --------------------------------------------------------

    def context_features(self, X, raw=None, raw_columns=None, train_mask=None):
        """
        Neighbourhood context for every node (no labels used).

        ``raw`` is an optional DataFrame (aligned with X) whose
        ``raw_columns`` are aggregated over neighbours and compared with
        the node's own value.

        With ``self.inductive`` the reference set is the training rows given by
        ``train_mask``; otherwise all rows are used.
        """
        X = np.asarray(X)
        if self.inductive:
            if train_mask is None:
                raise ValueError("inductive context features need train_mask")
            reference_rows = np.flatnonzero(np.asarray(train_mask, bool))
        else:
            reference_rows = np.arange(len(X))
        position = np.full(len(X), -1)
        position[reference_rows] = np.arange(len(reference_rows))

        distances, local = self._knn(X[reference_rows], X, exclude_self_rows=position)
        neighbours = reference_rows[local]

        out = {
            "knn_dist_first": distances[:, 0],
            "knn_dist_kth": distances[:, -1],
            "knn_dist_mean": distances.mean(axis=1),
        }

        if self.inductive:
            lof = LocalOutlierFactor(
                n_neighbors=self.k, metric=self.metric, n_jobs=self.n_jobs, novelty=True
            ).fit(X[reference_rows])
            score = -lof.score_samples(X)
            # Training rows: their own LOF within the training set (self excluded).
            score[reference_rows] = -lof.negative_outlier_factor_
            out["local_outlier_factor"] = score
        else:
            lof = LocalOutlierFactor(
                n_neighbors=self.k, metric=self.metric, n_jobs=self.n_jobs
            ).fit(X)
            out["local_outlier_factor"] = -lof.negative_outlier_factor_

        if raw is not None and raw_columns:
            values = raw[raw_columns].to_numpy(dtype=float)
            values = np.log1p(np.clip(values, 0, None))
            neighbour_mean = values[neighbours].mean(axis=1)
            neighbour_std = values[neighbours].std(axis=1)

            for j, col in enumerate(raw_columns):
                out[f"nbr_mean_{col}"] = neighbour_mean[:, j]
                out[f"nbr_std_{col}"] = neighbour_std[:, j]
                out[f"nbr_dev_{col}"] = values[:, j] - neighbour_mean[:, j]

        return pd.DataFrame(out)

    # --------------------------------------------------------
    # All features
    # --------------------------------------------------------

    def extract(
        self,
        X,
        labels,
        train_mask,
        families=None,
        raw=None,
        raw_columns=None,
    ):
        label_part = self.label_features(X, labels, train_mask, families)
        context_part = self.context_features(X, raw, raw_columns, train_mask=train_mask)

        return pd.concat([label_part, context_part], axis=1)

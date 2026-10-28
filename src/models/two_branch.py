"""
Family-gated detector.

Context features help some attack families and hurt others: host
behaviour lifts DDoS / Slowloris / infiltration detection on
CSE-CIC-IDS2018 but lowers almost every family on ToN-IoT. Instead of one
feature set for everything, this detector

1. trains one binary expert per feature set (e.g. flow, flow + host),
2. trains a router that predicts the attack family from flow features,
3. picks, per attack family and on VALIDATION data only, the expert with
   the best detection rate at the target false-positive rate,
4. maps every expert's score to a common scale - the share of validation
   benign traffic it exceeds - and mixes the chosen experts' scores using
   the router's family probabilities.

``ensemble_score`` (plain average of the calibrated experts) is provided as
a baseline, to show whether gating matters beyond simple ensembling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def _model(seed, balanced=True, max_iter=300, early_stopping="auto"):
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.1,
        class_weight="balanced" if balanced else None,
        early_stopping=early_stopping,
        random_state=seed,
    )


class FamilyGatedDetector:

    def __init__(self, target_fpr=0.01, seed=0, router_iterations=200):
        self.target_fpr = target_fpr
        self.seed = seed
        self.router_iterations = router_iterations

        self.experts_ = {}
        self.benign_reference_ = {}
        self.router_ = None
        self.choice_ = {}
        self.validation_tpr_ = None
        self.pooled_families_ = []

    # --------------------------------------------------------
    # Calibration to a common "benign percentile" scale
    # --------------------------------------------------------

    def _calibrate(self, name, scores):
        reference = self.benign_reference_[name]
        percentile = np.searchsorted(reference, scores, side="right") / len(reference)
        # Tiny monotone term keeps the ranking among flows above every
        # validation benign score instead of tying them all at 1.0.
        return percentile + 1e-3 * scores

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    def fit(self, expert_inputs, router_input, y, family, train_mask, val_mask,
            benign_label="Benign", min_validation_flows=50):
        """
        Parameters
        ----------
        expert_inputs : dict name -> array (n, d_e); the first entry is the
                        default expert when experts tie.
        router_input  : array (n, d) used to predict the family.
        y             : binary labels (1 = attack).
        family        : attack family per row (benign rows = ``benign_label``).
        train_mask, val_mask : boolean arrays.
        """
        y = np.asarray(y).astype(int)
        family = np.asarray(family).astype(str)
        train_mask = np.asarray(train_mask, bool)
        val_mask = np.asarray(val_mask, bool)
        self.benign_label = benign_label

        raw_val = {}
        for name, X in expert_inputs.items():
            expert = _model(self.seed).fit(X[train_mask], y[train_mask])
            self.experts_[name] = expert
            scores = expert.predict_proba(X[val_mask])[:, 1]
            raw_val[name] = scores
            self.benign_reference_[name] = np.sort(scores[y[val_mask] == 0])

        # No early stopping: its internal stratified split fails when a rare
        # family has a single training flow (e.g. SQL injection in some seeds).
        self.router_ = _model(
            self.seed, balanced=True, max_iter=self.router_iterations, early_stopping=False
        ).fit(router_input[train_mask], family[train_mask])

        # Per-family expert choice on validation data.
        val_family = family[val_mask]
        rows = []
        for name, scores in raw_val.items():
            threshold = np.quantile(scores[y[val_mask] == 0], 1 - self.target_fpr)
            for fam in np.unique(val_family[y[val_mask] == 1]):
                rows.append({
                    "family": fam,
                    "expert": name,
                    "val_tpr": float((scores[val_family == fam] > threshold).mean()),
                })
        table = pd.DataFrame(rows)
        self.validation_tpr_ = table.pivot(index="family", columns="expert", values="val_tpr")

        default = next(iter(expert_inputs))
        # Families with too few validation flows do not get a per-family choice:
        # they use the pooled default (the first expert), recorded in pooled_families_.
        val_counts = pd.Series(val_family[y[val_mask] == 1]).value_counts()
        self.pooled_families_ = sorted(f for f, c in val_counts.items() if c < min_validation_flows)
        for fam, row in self.validation_tpr_.iterrows():
            if fam in self.pooled_families_:
                self.choice_[fam] = default
                continue
            best = row.max()
            # Prefer the default (simplest) expert unless another is strictly better.
            self.choice_[fam] = default if row[default] >= best else row.idxmax()

        return self

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    def expert_scores(self, expert_inputs):
        return {
            name: self._calibrate(name, self.experts_[name].predict_proba(X)[:, 1])
            for name, X in expert_inputs.items()
        }

    def ensemble_score(self, expert_inputs):
        scores = self.expert_scores(expert_inputs)
        return np.mean(list(scores.values()), axis=0)

    def score(self, expert_inputs, router_input):
        calibrated = self.expert_scores(expert_inputs)
        default = next(iter(expert_inputs))

        proba = self.router_.predict_proba(router_input)
        classes = list(self.router_.classes_)
        attack_columns = [i for i, c in enumerate(classes) if c != self.benign_label]

        weights = proba[:, attack_columns]
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)

        combined = np.zeros(len(proba))
        for column, i in enumerate(attack_columns):
            expert = self.choice_.get(classes[i], default)
            combined += weights[:, column] * calibrated[expert]

        return combined

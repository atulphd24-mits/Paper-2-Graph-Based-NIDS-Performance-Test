"""Leakage-safe, imbalance-aware training utilities for threat classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class EvaluationResult:
    """Metrics and predictions from a single held-out test set."""

    metrics: dict[str, Any]
    report: pd.DataFrame
    predictions: pd.DataFrame


class ThreatClassifier:
    """Train and evaluate a multiclass threat classifier.

    The split happens before imputation, so statistics from the test set never
    influence training.  ``class_weight='balanced'`` counteracts the strong
    class imbalance in datasets such as CIC-IoT2023.
    """

    def __init__(
        self,
        label_column: str = "label",
        test_size: float = 0.2,
        random_state: int = 42,
        min_samples_per_class: int = 5,
        n_estimators: int = 300,
    ):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if min_samples_per_class < 2:
            raise ValueError("min_samples_per_class must be at least 2")

        self.label_column = label_column
        self.test_size = test_size
        self.random_state = random_state
        self.min_samples_per_class = min_samples_per_class
        self.n_estimators = n_estimators
        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = []
        self.excluded_classes: pd.Series | None = None

    def _prepare_data(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if self.label_column not in dataframe.columns:
            raise ValueError(f"Label column '{self.label_column}' was not found")

        target = dataframe[self.label_column]
        valid_rows = target.notna()
        data = dataframe.loc[valid_rows]
        target = target.loc[valid_rows]

        counts = target.value_counts()
        self.excluded_classes = counts[counts < self.min_samples_per_class]
        if not self.excluded_classes.empty:
            keep = target.isin(counts[counts >= self.min_samples_per_class].index)
            data = data.loc[keep]
            target = target.loc[keep]

        features = data.drop(columns=[self.label_column]).select_dtypes(include=[np.number])
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna(axis=1, how="all")
        if features.empty:
            raise ValueError("No usable numeric feature columns were found")
        if target.nunique() < 2:
            raise ValueError("At least two target classes are required")

        self.feature_columns = list(features.columns)
        return features, target

    def fit_evaluate(self, dataframe: pd.DataFrame) -> EvaluationResult:
        """Fit on a stratified training set and evaluate an untouched holdout."""
        features, target = self._prepare_data(dataframe)
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target,
        )

        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=self.n_estimators,
                        class_weight="balanced",
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        report_dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_precision": report_dict["macro avg"]["precision"],
            "macro_recall": report_dict["macro avg"]["recall"],
            "macro_f1": report_dict["macro avg"]["f1-score"],
            "weighted_f1": report_dict["weighted avg"]["f1-score"],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "classes_evaluated": target.nunique(),
        }
        predictions = pd.DataFrame(
            {"actual": y_test.to_numpy(), "predicted": y_pred}, index=y_test.index
        )
        report = pd.DataFrame(report_dict).transpose()
        return EvaluationResult(metrics=metrics, report=report, predictions=predictions)

    def feature_importance(self) -> pd.DataFrame:
        """Return impurity-based importance after ``fit_evaluate`` has run."""
        if self.pipeline is None:
            raise RuntimeError("Run fit_evaluate() before requesting feature importance")
        model = self.pipeline.named_steps["model"]
        return pd.DataFrame(
            {"feature": self.feature_columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)

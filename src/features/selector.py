"""
Hybrid Feature Selection Module
Author : Atul Chauhan
Project : SOC Threat Detection Framework
version : 1.3
"""

import pandas as pd
import numpy as np

from boruta import BorutaPy
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


class FeatureSelector:

    def __init__(self, X, y):
        self.X = X.copy()
        self.y = y.copy()

        self.selected_features = list(X.columns)

        self.correlation_scores = None
        self.mi_scores = None
        self.rf_scores = None

        self.log = []

    #########################################################
    # Correlation Filter
    #########################################################

    def correlation_filter(self, threshold=0.95):
        numeric_frame = self.X.select_dtypes(include=[np.number]).copy()

        if numeric_frame.shape[1] < 2:
            self.correlation_scores = pd.DataFrame(columns=["Feature1", "Feature2", "Correlation"])
            self.selected_features = list(self.X.columns)
            self.log.append("✓ Correlation Filter skipped because there were fewer than two numeric features")
            return []

        corr = numeric_frame.corr().abs()

        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        to_drop = []
        corr_pairs = []

        for column in upper.columns:
            correlated = upper.index[upper[column] > threshold].tolist()

            for other in correlated:
                corr_pairs.append({
                    "Feature1": other,
                    "Feature2": column,
                    "Correlation": upper.loc[other, column]
                })
                to_drop.append(column)

        self.correlation_scores = pd.DataFrame(corr_pairs)

        to_drop = list(dict.fromkeys(to_drop))

        self.X = self.X.drop(columns=to_drop)

        self.selected_features = list(self.X.columns)

        self.log.append(
            f"✓ Correlation Filter removed {len(to_drop)} features"
        )

        return to_drop

    #########################################################
    # Correlation Report
    #########################################################

    def correlation_report(self):
        if self.correlation_scores is None:
            print("Run correlation_filter() first.")
            return

        return self.correlation_scores.sort_values(
            by="Correlation",
            ascending=False
        )

    #########################################################
    # Mutual Information
    #########################################################

    def mutual_information(self):
        numeric_frame = self.X.select_dtypes(include=[np.number]).copy()

        if numeric_frame.shape[1] == 0:
            self.mi_scores = pd.DataFrame(columns=["Feature", "MI Score"])
            self.log.append("✓ Mutual Information skipped because there were no numeric features")
            return self.mi_scores

        mi = mutual_info_classif(
            numeric_frame,
            self.y,
            random_state=42
        )

        self.mi_scores = pd.DataFrame({
            "Feature": numeric_frame.columns,
            "MI Score": mi
        }).sort_values(
            by="MI Score",
            ascending=False
        )

        self.log.append("✓ Mutual Information completed")

        return self.mi_scores

    #########################################################
    # Random Forest Importance
    #########################################################

    def random_forest_importance(self):
        numeric_frame = self.X.select_dtypes(include=[np.number]).copy()

        if numeric_frame.shape[1] == 0:
            self.rf_scores = pd.DataFrame(columns=["Feature", "Importance"])
            self.log.append("✓ Random Forest Importance skipped because there were no numeric features")
            return self.rf_scores

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        model.fit(numeric_frame, self.y)

        self.rf_scores = pd.DataFrame({
            "Feature": numeric_frame.columns,
            "Importance": model.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=False
        )

        self.log.append(
            "✓ Random Forest Importance completed"
        )

        return self.rf_scores

    #########################################################
    # Boruta Selection
    #########################################################

    def boruta_selection(self, n_estimators='auto', perc=100, alpha=0.05):
        if self.rf_scores is None:
            self.random_forest_importance()

        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        feat_selector = BorutaPy(
            estimator=rf,
            n_estimators=n_estimators,
            verbose=0,
            random_state=42,
            alpha=alpha,
            perc=perc
        )

        numeric_frame = self.X.select_dtypes(include=[np.number]).copy()

        if numeric_frame.shape[1] == 0:
            self.boruta_scores = pd.DataFrame(columns=['Feature', 'Boruta Rank'])
            self.log.append('✓ Boruta selection skipped because there were no numeric features')
            return self.boruta_scores

        feat_selector.fit(numeric_frame.to_numpy(), self.y.to_numpy())

        self.boruta_scores = pd.DataFrame({
            'Feature': numeric_frame.columns,
            'Boruta Rank': feat_selector.ranking_
        })

        self.log.append(
            '✓ Boruta selection completed'
        )

        return self.boruta_scores

    #########################################################
    # Consensus Ranking
    #########################################################

    def combined_ranking(self):

        # Ensure prerequisite scores are available
        if self.mi_scores is None:
            self.mutual_information()

        if self.rf_scores is None:
            self.random_forest_importance()

        ranking = self.mi_scores.merge(
            self.rf_scores,
            on="Feature"
        )

        if hasattr(self, "boruta_scores"):

            ranking = ranking.merge(
                self.boruta_scores,
                on="Feature",
                how="left"
            )

        ranking["MI Rank"] = ranking["MI Score"].rank(
            ascending=False,
            method="dense"
        )

        ranking["RF Rank"] = ranking["Importance"].rank(
            ascending=False,
            method="dense"
        )

        if "Boruta Rank" in ranking.columns:

            ranking["Consensus Score"] = (
                ranking["MI Rank"] +
                ranking["RF Rank"] +
                ranking["Boruta Rank"]
            ) / 3

        else:

            ranking["Consensus Score"] = (
                ranking["MI Rank"] +
                ranking["RF Rank"]
            ) / 2

        ranking = ranking.sort_values(
            by="Consensus Score"
        )

        self.final_ranking = ranking

        return ranking

    #########################################################
    # Select Top Features
    #########################################################

    def select_top_features(self, top_n=20):

        ranking = self.final_ranking.copy()

        ranking["Selected"] = "No"

        ranking.loc[
            ranking.index[:top_n],
            "Selected"
        ] = "Yes"

        return ranking

    #########################################################
    # Summary
    #########################################################

    def summary(self):
        print("=" * 60)
        print("FEATURE SELECTION REPORT")
        print("=" * 60)

        print()
        print(f"Remaining Features : {len(self.X.columns)}")
        print()

        for item in self.log:
            print(item)
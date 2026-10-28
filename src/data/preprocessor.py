"""
Data Preprocessor Module
Author : Atul Chauhan
Project: SOC Threat Detection Framework
"""

import pandas as pd
from sklearn.feature_selection import VarianceThreshold


class DataPreprocessor:
    """
    Generic data preprocessing class.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str):
        self.df = df.copy()
        self.dataset_name = dataset_name

        self.initial_shape = self.df.shape
        self.final_shape = None

        self.log = []

    def standardize_columns(self):
        """
        Standardize column names.
        """

        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        self.log.append("✓ Standardized column names")

    def remove_duplicates(self):
        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        removed = before - len(self.df)

        self.log.append(f"✓ Duplicate rows removed : {removed}")

    def remove_constant_features(self):
        constant_columns = [
            col
            for col in self.df.columns
            if self.df[col].nunique() <= 1
        ]

        self.df.drop(columns=constant_columns, inplace=True)

        self.log.append(
            f"✓ Constant features removed : {len(constant_columns)}"
        )

        self.log.append(
            f"Removed : {constant_columns}"
        )

    def remove_low_variance_features(self, threshold=0.0):
        numeric = self.df.select_dtypes(include="number")

        if numeric.empty:
            self.log.append("✓ No numeric features available for variance filtering")
            return

        selector = VarianceThreshold(threshold)

        selector.fit(numeric)

        keep = numeric.columns[selector.get_support()]

        removed = list(set(numeric.columns) - set(keep))

        self.df = pd.concat(
            [
                self.df.drop(columns=numeric.columns),
                numeric[keep]
            ],
            axis=1,
        )

        self.log.append(
            f"✓ Low variance features removed : {len(removed)}"
        )

    def preprocess(self):
        self.standardize_columns()

        self.remove_duplicates()

        self.remove_constant_features()

        self.remove_low_variance_features()

        self.final_shape = self.df.shape

        return self.df

    def generate_log(self):
        print("=" * 60)

        print("DATA PREPROCESSING REPORT")

        print("=" * 60)

        print(f"Dataset : {self.dataset_name}")

        print()

        print(f"Initial Shape : {self.initial_shape}")

        print(f"Final Shape   : {self.final_shape}")

        print()

        for item in self.log:
            print(item)
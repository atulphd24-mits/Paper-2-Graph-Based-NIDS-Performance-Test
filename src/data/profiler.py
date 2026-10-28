"""
profiler.py

Professional Dataset Profiler
SOC Threat Detection Framework
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class DatasetProfiler:
    """Profile a dataset and generate summary reports/plots."""

    def __init__(self, dataframe, dataset_name="Dataset", label_column="label", attack_column=None):
        self.df = dataframe.copy()
        self.dataset_name = dataset_name
        self.label_column = label_column
        self.attack_column = attack_column

    def shape(self):
        return self.df.shape

    def memory_usage(self):
        return round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)

    def missing_values(self):
        return self.df.isnull().sum()

    def duplicate_rows(self):
        return self.df.duplicated().sum()

    def data_types(self):
        return self.df.dtypes

    def class_distribution(self):
        if self.label_column not in self.df.columns:
            return None
        return self.df[self.label_column].value_counts()

    def constant_features(self):
        return [col for col in self.df.columns if self.df[col].nunique() == 1]

    def low_variance_features(self, threshold=0.001):
        numeric = self.df.select_dtypes(include="number")
        variances = numeric.var()
        return variances[variances < threshold]

    def summary(self):
        report = {
            "Label_Column": self.label_column,
            "Attack_Column": self.attack_column,
            "Dataset": self.dataset_name,
            "Rows": self.df.shape[0],
            "Columns": self.df.shape[1],
            "Memory_MB": self.memory_usage(),
            "Missing_Values": int(self.df.isnull().sum().sum()),
            "Duplicate_Rows": int(self.duplicate_rows()),
            "Numeric_Features": len(self.df.select_dtypes(include="number").columns),
            "Categorical_Features": len(self.df.select_dtypes(exclude="number").columns),
            "Constant_Features": len(self.constant_features()),
            "Low_Variance": len(self.low_variance_features()),
        }
        return report

    def print_report(self):
        report = self.summary()
        print("=" * 60)
        print("DATASET PROFILE")
        print("=" * 60)
        for key, value in report.items():
            print(f"{key:25}: {value}")
        print("=" * 60)

    def save_report(self, output_file):
        report = pd.DataFrame([self.summary()])
        report.to_csv(output_file, index=False)

    def plot_class_distribution(self, save_path=None, show=True):
        """Plot class distribution."""
        if self.label_column not in self.df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found.")

        counts = self.df[self.label_column].value_counts()
        plt.figure(figsize=(10, 5))
        counts.plot(kind="bar")
        plt.title(f"Class Distribution ({self.label_column})")
        plt.xlabel(self.label_column)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

        return counts

    def attack_distribution(self):
        if self.attack_column is None:
            return None
        if self.attack_column not in self.df.columns:
            return None
        return self.df[self.attack_column].value_counts()

    def plot_attack_distribution(self, save_path=None, show=True):
        if self.attack_column is None:
            raise ValueError("Attack column is not configured.")
        if self.attack_column not in self.df.columns:
            raise ValueError(f"Attack column '{self.attack_column}' not found.")

        counts = self.df[self.attack_column].value_counts()
        plt.figure(figsize=(12, 6))
        counts.plot(kind="bar")
        plt.title("Attack Distribution")
        plt.xlabel(self.attack_column)
        plt.ylabel("Count")
        plt.xticks(rotation=90)
        plt.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

        return counts

    def plot_missing_values(self, save_path=None, show=True):
        missing = self.df.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]

        if missing.empty:
            return missing

        plt.figure(figsize=(12, 6))
        missing.plot(kind="bar")
        plt.title("Missing Values")
        plt.xlabel("Feature")
        plt.ylabel("Missing Count")
        plt.xticks(rotation=90)
        plt.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

        return missing

    def plot_correlation(self, save_path=None, show=True):
        numeric = self.df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return None

        corr = numeric.corr()
        plt.figure(figsize=(12, 8))
        plt.imshow(corr, cmap="coolwarm")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Feature Correlation")
        plt.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

        return corr

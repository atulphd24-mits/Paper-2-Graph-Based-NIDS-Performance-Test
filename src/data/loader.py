"""
Generic Dataset Loader for the SOC Threat Detection Framework

Author  : Atul Chauhan
Project : SOC-Threat-Detection
"""

from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class DatasetLoader:
    """
    Generic Dataset Loader

    Supports
    --------
    ✓ Single CSV
    ✓ Multiple CSV
    ✓ Single Parquet
    ✓ Multiple Parquet
    ✓ Recursive Folder Search
    ✓ Iterator Mode
    """

    def __init__(self, dataset_path: str):

        project_root = Path(__file__).resolve().parents[2]

        path = Path(dataset_path)

        if not path.is_absolute():
            path = project_root / path

        self.dataset_path = path

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"{self.dataset_path} does not exist."
            )

    # ---------------------------------------------------------
    # Find dataset files
    # ---------------------------------------------------------
    def list_files(self):

        files = []

        if self.dataset_path.is_file():

            if self.dataset_path.suffix.lower() in [
                ".csv",
                ".parquet"
            ]:
                files.append(self.dataset_path)

        else:

            files.extend(
                sorted(self.dataset_path.rglob("*.csv"))
            )

            files.extend(
                sorted(self.dataset_path.rglob("*.parquet"))
            )

        if len(files) == 0:
            raise FileNotFoundError(
                f"No CSV or Parquet files found in {self.dataset_path}"
            )

        logging.info(
            f"Found {len(files)} dataset file(s)."
        )

        return files

    def list_csv_files(self):
        """
        Return only CSV files from the dataset path (compatibility helper).
        """

        files = self.list_files()

        csv_files = [f for f in files if f.suffix.lower() == ".csv"]

        logging.info(f"Found {len(csv_files)} CSV file(s).")

        return csv_files

    def load_csv(self, file_ref, columns=None):
        """
        Compatibility wrapper to load a CSV file.

        Accepts a `Path`/string pointing to a CSV file, or an integer
        index referencing the list returned by `list_csv_files()`.
        """

        # If the user passed an integer index, resolve to a file path
        if isinstance(file_ref, int):
            files = self.list_csv_files()
            try:
                file_path = files[file_ref]
            except IndexError as exc:
                raise IndexError("CSV file index out of range") from exc

        else:
            file_path = Path(file_ref)

            if not file_path.is_absolute():
                file_path = self.dataset_path / file_path

        if file_path.suffix.lower() != ".csv":
            raise ValueError("load_csv only supports CSV files")

        return self.load_file(file_path, columns=columns)

    def list_parquet_files(self):
        """
        Return only Parquet files from the dataset path (compatibility helper).
        """

        files = self.list_files()

        parquet_files = [f for f in files if f.suffix.lower() == ".parquet"]

        logging.info(f"Found {len(parquet_files)} Parquet file(s).")

        return parquet_files

    def load_parquet(self, file_ref, columns=None):
        """
        Compatibility wrapper to load a Parquet file.

        Accepts a `Path`/string pointing to a Parquet file, or an integer
        index referencing the list returned by `list_parquet_files()`.
        """

        # If the user passed an integer index, resolve to a file path
        if isinstance(file_ref, int):
            files = self.list_parquet_files()
            try:
                file_path = files[file_ref]
            except IndexError as exc:
                raise IndexError("Parquet file index out of range") from exc

        else:
            file_path = Path(file_ref)

            if not file_path.is_absolute():
                file_path = self.dataset_path / file_path

        if file_path.suffix.lower() != ".parquet":
            raise ValueError("load_parquet only supports Parquet files")

        return self.load_file(file_path, columns=columns)

    # ---------------------------------------------------------
    # Load one file
    # ---------------------------------------------------------
    def load_file(self, file_path: Path, columns=None):

        logging.info(f"Loading : {file_path.name}")

        suffix = file_path.suffix.lower()

        if suffix == ".csv":

            return pd.read_csv(
                file_path,
                usecols=columns
            )

        elif suffix == ".parquet":

            return pd.read_parquet(
                file_path,
                columns=columns
            )

        else:

            raise ValueError(
                f"Unsupported format : {suffix}"
            )

    # ---------------------------------------------------------
    # Load entire dataset
    # ---------------------------------------------------------
    def load_all(self, file_path: Path = None, columns=None, sample_fraction: float = 1.0):

        files = self.list_files()

        dfs = []

        for file in files:

            df = self.load_file(
                file,
                columns=columns
            )

            # Sample before retaining the frame so large multi-file datasets
            # do not accumulate the full dataset in memory.
            if sample_fraction < 1.0 and len(df) > 0:
                df = df.sample(
                    frac=sample_fraction,
                    random_state=42
                )

            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        dataset = pd.concat(
            dfs,
            ignore_index=True
        )

        logging.info(
            f"Dataset Shape : {dataset.shape}"
        )

        return dataset

    # ---------------------------------------------------------
    # Iterate over dataset
    # ---------------------------------------------------------
    def iterate(self, columns=None):

        files = self.list_files()

        for file in files:

            yield (
                file.name,
                self.load_file(
                    file,
                    columns=columns
                )
            )

    # ---------------------------------------------------------
    # Dataset Summary
    # ---------------------------------------------------------
    def summary(self):

        files = self.list_files()

        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        print(f"Location : {self.dataset_path}")
        print(f"Files    : {len(files)}")

        csv_count = sum(
            f.suffix.lower() == ".csv"
            for f in files
        )

        parquet_count = sum(
            f.suffix.lower() == ".parquet"
            for f in files
        )

        print(f"CSV Files      : {csv_count}")
        print(f"Parquet Files  : {parquet_count}")

        print("=" * 60)
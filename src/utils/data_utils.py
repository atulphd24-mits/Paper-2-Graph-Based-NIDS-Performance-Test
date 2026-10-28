from src.utils.config import Config
from src.data.loader import DatasetLoader


def load_dataset(dataset_name):
    config = Config()

    dataset_path = config.get_dataset_path(dataset_name)

    loader = DatasetLoader(dataset_path)

    csv_files = loader.list_csv_files()

    if csv_files:
        return loader.load_csv(csv_files[0])

    parquet_files = loader.list_parquet_files()

    if parquet_files:
        return loader.load_parquet(parquet_files[0])

    raise FileNotFoundError(
        f"No CSV or Parquet files found in {dataset_path}"
    )
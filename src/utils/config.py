"""
config.py

Configuration Manager
SOC Threat Detection Framework
"""

from pathlib import Path
import yaml


class Config:

    def __init__(self, config_file=None):

        # Find project root automatically
        project_root = Path(__file__).resolve().parents[2]

        if config_file is None:
            self.config_file = project_root / "configs" / "config.yaml"
        else:
            candidate = Path(config_file)
            if not candidate.is_absolute():
                candidate = project_root / candidate
            self.config_file = candidate

        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found:\n{self.config_file}"
            )

        with open(self.config_file, "r") as file:
            self.config = yaml.safe_load(file)

    def get_dataset_path(self, dataset_name):
        return self.config["datasets"][dataset_name]["path"]

    def get_label_column(self, dataset_name):
        return self.config["datasets"][dataset_name]["label"]

    def get_output_path(self, key):
        return self.config["output"][key]
import yaml
import pandas as pd

__all__ = ["load_config", "get_num_classes"]


def load_config(config_file_path: str) -> dict:
    """
    Load the configuration file.

    Args:
        config_file_path (str): The path to the configuration file

    Returns:
        dict: Loaded configuration
    """
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_num_classes(csv_path):
    """Get the number of unique classes from the CSV file"""
    df = pd.read_csv(csv_path)
    return len(df["label"].unique())

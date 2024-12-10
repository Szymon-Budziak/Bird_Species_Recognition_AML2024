import yaml
import pandas as pd


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
    num_classes = len(df["label"].unique())
    print(f"Number of classes: {num_classes}")
    return num_classes


def unfreeze_layers(model, num_layers: int = 5):
    for param in model.parameters():
        param.requires_grad = True

    for param in list(model.parameters())[-num_layers:]:
        param.requires_grad = True

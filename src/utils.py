import yaml
import pandas as pd
import subprocess
import os
import torch


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


def save_submission(submission, submission_path):
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    pd.DataFrame(submission, columns=["id", "label"]).to_csv(
        submission_path, index=False
    )
    print(f"Submission saved to {submission_path}")


def submit_to_kaggle(file_path, message):
    command = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        "aml-2024-feather-in-focus",
        "-f",
        file_path,
        "-m",
        message,
    ]
    try:
        subprocess.run(command, check=True)
        print("Submission successful!")
    except subprocess.CalledProcessError as e:
        print("Submission failed:", e.stderr)


def save_best_model(model, optimizer, scheduler, epoch, best_acc, model_path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
    }, model_path)

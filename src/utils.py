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


def save_best_model(model, optimizer, scheduler, best_acc, epoch, best_model_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "epoch": epoch,
        },
        best_model_path,
    )
    print(f"Saved new best model with accuracy: {best_acc:.4f}")

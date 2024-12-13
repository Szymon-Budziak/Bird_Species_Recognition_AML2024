import yaml
import pandas as pd
import numpy as np
import subprocess
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def save_submission(submission: pd.DataFrame, submission_path: str) -> None:
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    pd.DataFrame(submission, columns=["id", "label"]).to_csv(
        submission_path, index=False
    )
    print(f"Submission saved to {submission_path}")


def submit_to_kaggle(file_path: str, message: str) -> None:
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


def save_best_model(
    model, optimizer, scheduler, best_acc, epoch, config, best_model_path
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "epoch": epoch,
            "config": config,
        },
        best_model_path,
    )
    print(f"Saved new best model with accuracy: {best_acc:.4f}")


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, np.long)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        # Reshape target
        target = target.view(-1, 1)

        # Apply log_softmax with explicit dimension
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        # Apply alpha if specified
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        # Calculate focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt

        # Return mean or sum based on size_average
        return loss.mean() if self.size_average else loss.sum()


def create_final_submission(best_submission_path: str) -> pd.DataFrame:
    """
    Create the final submission by averaging the best submissions.
    """
    best_submissions_list = os.listdir(best_submission_path)
    best_submissions_df = pd.DataFrame()

    for idx, submission in enumerate(best_submissions_list):
        submission_path = os.path.join(best_submission_path, submission)
        submission_df = pd.read_csv(submission_path).set_index("id")
        submission_df.rename(columns={"label": f"label_{idx}"}, inplace=True)
        best_submissions_df = pd.concat([best_submissions_df, submission_df], axis=1)

    best_submissions_df["final_label"] = best_submissions_df.apply(
        lambda x: x.value_counts().idxmax(), axis=1
    )

    final_submission_df = pd.DataFrame(
        data={
            "id": best_submissions_df.index,
            "label": best_submissions_df["final_label"],
        }
    ).reset_index(drop=True)

    final_submission_df.to_csv("final_submission.csv", index=False)

    return final_submission_df

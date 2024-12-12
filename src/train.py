import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from utils import load_config
import os
from collections import Counter

from custom_models import ModelFactory
from utils import save_submission, submit_to_kaggle, save_best_model
from data_generator import get_data_loaders


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    num: int,
    config: dict,
):
    print("Training model")

    # Create directories for models and submissions
    os.makedirs(config["models_dir"], exist_ok=True)
    os.makedirs(config["submission_dir"], exist_ok=True)

    best_model_path = os.path.join(
        config["models_dir"], f"best_model_{config['model_type']}_{num}.pt"
    )
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = val_loader

            running_loss = running_corrects = total = 0.0

            # Iterate over data
            for images, labels, attributes in dataloader:
                images, labels, attributes = (
                    images.to(device),
                    labels.to(device),
                    attributes.to(device),
                )

                # Zero the parameter gradients
                if phase == "train":
                    optimizer.zero_grad()

                # Forward + Track history if only in train phase
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images, attributes)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                total += labels.size(0)
                running_corrects += preds.eq(labels).sum().item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = 100 * running_corrects / total

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    save_best_model(
                        model, optimizer, scheduler, best_acc, epoch, best_model_path
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1

        # Check early stopping
        if patience_counter >= config["patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\nTraining complete")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate_model(model, test_loader: DataLoader, device: torch.device) -> list:
    print("Evaluating model")
    model.eval()
    submission = []

    with torch.no_grad():
        for images, ids, attributes in test_loader:
            images = images.to(device)
            attributes = attributes.to(device)

            outputs = model(images, attributes)
            _, preds = torch.max(outputs, 1)

            preds_np = preds.cpu().numpy() + 1

            if isinstance(ids, torch.Tensor):
                ids_np = ids.cpu().numpy()
            else:
                ids_np = np.array(ids)

            submission.extend(zip(ids_np, preds_np))

    return submission


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1)
    args = parser.parse_args()

    config = load_config("configs/config.yaml")

    attributes = torch.tensor(np.load(config["attributes_path"]), dtype=torch.float32)
    train_df = pd.read_csv(config["train_csv_path"])
    test_df = pd.read_csv(config["test_csv_path"])

    train_loader, val_loader, test_loader = get_data_loaders(
        config, train_df, test_df, attributes
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ModelFactory.create_model(
        config["model_type"], config["num_classes"], config["attribute_dim"]
    ).to(device)

    # Optimizer, criterion and scheduler
    class_counts = Counter(train_df["label"])
    class_weights = np.array(
        [1.0 / class_counts.get(i, 1) for i in range(config["num_classes"])]
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=config["label_smoothing"]
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train
    best_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        args.num,
        config,
    )

    # Evaluate
    submission = evaluate_model(best_model, test_loader, device)
    submission_path = os.path.join(
        config["submission_dir"], f"submission_{args.num}.csv"
    )
    save_submission(submission, submission_path)
    submit_to_kaggle(submission_path, f"Submission {args.num}")

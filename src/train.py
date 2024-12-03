import os
from tempfile import TemporaryDirectory
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler

from process_data import get_data_loaders
from utils import load_config, get_num_classes


def train_model(
    model, criterion, optimizer, scheduler, num_epochs, train_loader, val_loader, device
):
    best_acc = 0.0
    best_model_state = None

    with TemporaryDirectory() as temp_dir:
        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                    dataloader = train_loader
                else:
                    model.eval()  # Set model to evaluate mode
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward pass + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # Deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_state = model.state_dict()

        # Load best model state
        model.load_state_dict(best_model_state)

    return model


if __name__ == "__main__":
    # Load preprocessing configuration
    preprocessing_config = load_config("configs/config.yaml")

    # Prepare data loaders
    train_loader, val_loader, test_loader = get_data_loaders(preprocessing_config)

    # Load pre-trained ResNet model
    model = models.resnet50(weights="IMAGENET1K_V2")
    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classifier
    num_ftrs = model.fc.in_features
    num_classes = get_num_classes(preprocessing_config["train_csv_path"])
    print(f"Number of classes: {num_classes}")
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=exp_lr_scheduler,
        num_epochs=25,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Save the final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": num_classes,
            "class_to_idx": (
                train_loader.dataset.class_to_idx
                if hasattr(train_loader.dataset, "class_to_idx")
                else None
            ),
        },
        "bird_classifier.pth",
    )

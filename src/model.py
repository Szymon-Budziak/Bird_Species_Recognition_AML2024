import os
import torch
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from PIL import Image

# --------------------------
# Dataset Definitions
# --------------------------

class BirdTrainDataset(Dataset):
    """
    Custom Dataset for loading bird images with their labels and attributes.
    """

    def __init__(self, csv_file: str, images_dir: str, attributes_file: str, transform=None) -> None:
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

        # Convert 1-based labels to 0-based indices
        self.unique_labels = sorted(self.data["label"].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}

        # Load attributes
        self.attributes = np.load(attributes_file, allow_pickle=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Image path and species label
        img_name = self.data.iloc[idx, 0]
        if img_name.startswith("/"):
            img_name = img_name[1:]
        img_path = os.path.join(self.images_dir, img_name)

        original_label = self.data.iloc[idx, 1]
        label = self.label_to_idx[original_label]  # Convert to 0-based index

        # Attribute labels (from the attribute matrix)
        attribute_label = self.attributes[label]

        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, torch.tensor(attribute_label, dtype=torch.float32)


# --------------------------
# Model Definition
# --------------------------

class BirdMultiTaskModel(nn.Module):
    def __init__(self, num_species, num_attributes):
        super(BirdMultiTaskModel, self).__init__()
        # Backbone feature extractor
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove default FC layer

        # Heads
        self.species_head = nn.Linear(in_features, num_species)  # Species classifier
        self.attribute_head = nn.Linear(in_features, num_attributes)  # Attribute predictor

    def forward(self, x):
        features = self.backbone(x)
        species_logits = self.species_head(features)
        attribute_logits = self.attribute_head(features)
        return species_logits, attribute_logits


# --------------------------
# DataLoader and Transforms
# --------------------------

def get_transforms(train=True):
    """
    Get transformations for training or validation/test datasets.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def get_data_loaders(train_csv, train_dir, attributes_file, batch_size=32, num_workers=4):
    """
    Create DataLoaders for training and validation.
    """
    full_train_dataset = BirdTrainDataset(
        csv_file=train_csv,
        images_dir=train_dir,
        attributes_file=attributes_file,
        transform=get_transforms(train=True),
    )

    # Train-validation split
    val_size = int(0.15 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# --------------------------
# Training Loop
# --------------------------

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    # Loss functions
    species_criterion = nn.CrossEntropyLoss()
    attribute_criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_species_correct = 0
        train_total_samples = 0

        for images, species_labels, attribute_labels in train_loader:
            images, species_labels, attribute_labels = images.to(device), species_labels.to(device), attribute_labels.to(device)

            # Forward pass
            species_logits, attribute_logits = model(images)

            # Compute losses
            species_loss = species_criterion(species_logits, species_labels)
            attribute_loss = attribute_criterion(attribute_logits, attribute_labels)

            # Combined loss
            loss = species_loss + 0.5 * attribute_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Species accuracy
            species_preds = species_logits.argmax(dim=1)
            train_species_correct += (species_preds == species_labels).sum().item()
            train_total_samples += species_labels.size(0)

        train_species_accuracy = train_species_correct / train_total_samples
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_species_accuracy:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_species_correct = 0
        val_total_samples = 0
        attribute_precisions = []

        with torch.no_grad():
            for images, species_labels, attribute_labels in val_loader:
                images, species_labels, attribute_labels = (
                    images.to(device),
                    species_labels.to(device),
                    attribute_labels.to(device),
                )

                species_logits, attribute_logits = model(images)

                # Compute losses
                species_loss = species_criterion(species_logits, species_labels)
                attribute_loss = attribute_criterion(attribute_logits, attribute_labels)

                loss = species_loss + 0.5 * attribute_loss
                val_loss += loss.item()

                # Species accuracy
                species_preds = species_logits.argmax(dim=1)
                val_species_correct += (species_preds == species_labels).sum().item()
                val_total_samples += species_labels.size(0)

                # Attribute precision
                attribute_preds = (torch.sigmoid(attribute_logits) > 0.5).float()
                sample_precisions = (
                    (attribute_preds == attribute_labels).float().mean(dim=1).cpu().numpy()
                )
                attribute_precisions.extend(sample_precisions)

        val_species_accuracy = val_species_correct / val_total_samples
        val_attribute_precision = np.mean(attribute_precisions)

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, "
              f"Validation Species Accuracy: {val_species_accuracy:.4f}, "
              f"Validation Attribute Precision: {val_attribute_precision:.4f}")


# --------------------------
# Main Script
# --------------------------

if __name__ == "__main__":
    # Configurations
    train_csv = "data/train_images.csv"
    train_dir = "data/train_images"
    attributes_file = "data/attributes.npy"
    batch_size = 32
    num_epochs = 15
    learning_rate = 1e-4
    num_species = 200
    num_attributes = 312


    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get DataLoaders
    train_loader, val_loader = get_data_loaders(train_csv, train_dir, attributes_file, batch_size=batch_size)

    # Model
    model = BirdMultiTaskModel(num_species=num_species, num_attributes=num_attributes).to(device)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)


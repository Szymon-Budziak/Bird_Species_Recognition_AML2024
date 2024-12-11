import os
import torch
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

__all__ = ["get_data_loaders", "create_submission_file"]

# Load class names
class_names = np.load("data/class_names.npy", allow_pickle=True).item()


class BirdTrainDataset(Dataset):
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
    

class BirdTestDataset(Dataset):
    """
    Custom Dataset for loading test bird images (no labels).
    """

    def __init__(self, csv_file: str, images_dir: str, transform=None) -> None:
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.image_ids = self.data["id"].values

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx]["image_path"]
        if img_name.startswith("/"):
            img_name = img_name[1:]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, self.image_ids[idx]


def get_transforms(train=True, config: dict = None):
    """
    Get transformations for training or validation/test datasets.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize(size=tuple((config["resize_size"]))),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(25),
                transforms.ColorJitter(
                    brightness=config["brightness"],
                    contrast=config["contrast"],
                    saturation=config["saturation"],
                    hue=config["hue"],
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(size=tuple(config["resize_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def get_data_loaders(config: dict) -> tuple:
    """
    Create DataLoaders for training, validation, and testing.
    """
    full_train_dataset = BirdTrainDataset(
        csv_file=config["train_csv_path"],
        images_dir=config["train_dir_path"],
        transform=get_transforms(train=True, config=config),
    )

    # Total dataset size
    total_dataset_size = len(full_train_dataset)
    val_size = int(total_dataset_size * config.get("val_size", 0.15))
    train_size = total_dataset_size - val_size
    print(
        f"Total training dataset size: {total_dataset_size} | Training size: {train_size} | Validation size: {val_size}"
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    test_dataset = BirdTestDataset(
        csv_file=config["test_csv_path"],
        images_dir=config["test_dir_path"],
        transform=get_transforms(train=False, config=config),
    )

    print(f"Total test dataset size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    return train_loader, val_loader, test_loader


def create_submission_file(model, test_loader, output_csv):
    """
    Generate submission CSV file with predictions for species and attributes.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(next(model.parameters()).device)
            species_logits, attribute_logits = model(images)

            # Species predictions
            species_preds = species_logits.argmax(dim=1).cpu().numpy()

            # Attribute predictions (using a threshold, e.g., 0.5)
            attribute_preds = (torch.sigmoid(attribute_logits) > 0.5).int().cpu().numpy()

            for img_id, sp, attr in zip(image_ids, species_preds, attribute_preds):
                results.append({
                    "id": img_id,
                    "species": sp,
                    "attributes": list(attr)  # Convert numpy array to list for JSON-friendly format
                })

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Submission file saved to {output_csv}")

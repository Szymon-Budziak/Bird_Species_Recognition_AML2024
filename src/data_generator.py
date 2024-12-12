from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision import transforms
import random


class BirdDataset(Dataset):
    """
    Custom Dataset for loading bird images with their labels and attributes
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        img_dir: str,
        attributes: torch.Tensor,
        transform: transforms.Compose = None,
        is_train: bool = True,
    ):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.attributes_tensor = attributes
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.is_train:
            img_name = str(self.dataframe.iloc[idx, 0]).lstrip("/")
            img_path = os.path.join(self.img_dir, img_name)
            label = self.dataframe.iloc[idx, 1] - 1  # Convert to 0-indexed
            attributes = self.attributes_tensor[label]
        else:
            img_name = str(self.dataframe.iloc[idx, 1]).lstrip("/")
            img_path = os.path.join(self.img_dir, img_name)
            label = str(self.dataframe.iloc[idx, 0])  # in this case it is id
            attributes = torch.zeros(self.attributes_tensor.size(1))

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, attributes


def get_transforms(
    config: dict,
) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """
    Get the transformations for the train, validation, and test datasets.
    """
    train_transformations = [
        transforms.Resize(size=tuple(config["resize_size"])),
        transforms.RandomRotation(10),
        # transforms.RandomAffine(
        #     0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=20
        # ),
        transforms.ToTensor(),
    ]

    test_transformations = [
        transforms.Resize(size=tuple(config["resize_size"])),
        transforms.ToTensor(),
    ]

    if config["model_type"] != "EFF_NET_B2":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_transformations.append(normalize)
        test_transformations.append(normalize)

    train_transform = transforms.Compose(train_transformations)
    test_transform = transforms.Compose(test_transformations)

    # Validation transform is the same as test transform
    return train_transform, test_transform, test_transform


def get_data_loaders(
    config: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    attributes: torch.Tensor,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.
    """
    total_size = len(train_df)

    # Split the train_df into train and validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=config["val_size"],
        shuffle=True,
        random_state=101,
        stratify=train_df["label"] if "label" in train_df.columns else None,
    )
    print(
        f"Total train dataset size: {total_size} | Training size: {len(train_df)} | Validation size: {len(val_df)} | Test size: {len(test_df)}"
    )

    train_transform, val_transform, test_transform = get_transforms(config)

    train_dataset = BirdDataset(
        train_df, config["train_dir_path"], attributes, train_transform
    )
    val_dataset = BirdDataset(
        val_df, config["train_dir_path"], attributes, val_transform
    )
    test_dataset = BirdDataset(
        test_df, config["test_dir_path"], attributes, test_transform, is_train=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def augment_data(image: torch.Tensor) -> torch.Tensor:
    """
    Augment the image with random transformations.
    """
    if random.random() > 0.75:
        image = transforms.functional.hflip(image)

    if random.random() >= 0.4:
        image = transforms.functional.adjust_saturation(
            image, saturation_factor=random.uniform(0.7, 1.3)
        )

    if random.random() >= 0.4:
        image = transforms.functional.adjust_contrast(
            image, contrast_factor=random.uniform(0.8, 1.2)
        )

    if random.random() >= 0.4:
        image = transforms.functional.adjust_brightness(
            image, brightness_factor=random.uniform(0.9, 1.1)
        )

    return image

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


def augment_data(image):
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


class BirdDataset(Dataset):
    """
    Custom Dataset for loading bird images with their labels.
    """

    def __init__(self, dataframe, transform=None, is_test=False):
        self.dataframe = dataframe
        self.transform = transform
        self.is_test = is_test
        self.df_labels = self.dataframe["label"]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx]["image_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image
        else:
            label = self.dataframe.iloc[idx]["label"]
            return image, label


class DataGenerator:
    def __init__(self, config: dict):
        self.config = config

    def generate_train_val_data(self, model_type: str):
        if model_type == "EFF_NET_B2":
            train_transform = transforms.Compose(
                [
                    transforms.Resize((self.config["resize_size"])),
                    transforms.RandomRotation(self.config["rotation"]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(
                        0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=20
                    ),
                    transforms.ToTensor(),
                ]
            )

            val_transform = transforms.Compose(
                [
                    transforms.Resize((self.config["resize_size"])),
                    transforms.ToTensor(),
                ]
            )
        elif model_type == "VIT":
            train_transform = transforms.Compose(
                [
                    transforms.Resize((self.config["resize_size"])),
                    transforms.RandomRotation(110),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(
                        0, translate=(0.2, 0.2), scale=(0.4, 1.6), shear=15
                    ),
                    transforms.Lambda(lambda x: augment_data(x)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            val_transform = transforms.Compose(
                [
                    transforms.Resize((self.config["resize_size"])),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        return self.__generate_train_val_data_loader(train_transform, val_transform)

    def generate_test_data(self, model_type):
        if model_type == "EFF_NET_B2":
            transform = transforms.Compose(
                [
                    transforms.Resize((self.config["resize_size"])),
                    transforms.ToTensor(),
                ]
            )
        elif model_type == "VIT":
            transform = transforms.Compose(
                [
                    transforms.Resize(self.config["resize_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        return self.__generate_test_data_loader(transform)

    def __generate_train_val_data_loader(self, train_transform, val_transform):
        # Load class names
        bird_type_classes = np.load(
            self.config["class_names_path"], allow_pickle=True
        ).item()
        bird_type_classes_swapped = {
            value: key for key, value in bird_type_classes.items()
        }

        # Load and prepare training data
        train_img_df = pd.read_csv(self.config["train_csv_path"])
        train_img_df["class_name"] = train_img_df["label"].map(
            bird_type_classes_swapped
        )
        train_img_df["image_path"] = train_img_df["image_path"].apply(
            lambda x: f"{self.config['train_dir_path']}/{x[1:]}"
        )

        train_df, val_df = train_test_split(
            train_img_df,
            train_size=0.80,
            shuffle=True,
            random_state=124,
            stratify=train_img_df["label"],
        )

        train_dataset = BirdDataset(train_df, transform=train_transform)
        val_dataset = BirdDataset(val_df, transform=val_transform)
        print(f"Total train dataset size: {len(train_dataset)}")
        print(f"Total validation dataset size: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

        train_loader.dataset.targets = train_img_df["class_name"]

        return train_loader, val_loader

    def __generate_test_data_loader(self, transform):
        test_images_df = pd.read_csv(self.config["test_csv_path"])
        test_images_df["image_path"] = test_images_df["image_path"].apply(
            lambda x: f"{self.config['test_dir_path']}/{x[1:]}"
        )

        test_dataset = BirdDataset(test_images_df, transform=transform, is_test=True)
        print(f"Total test dataset size: {len(test_dataset)}")

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

        return test_loader

import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision import models


class ModelFactory:
    @staticmethod
    def create_model(model_type, num_classes, attribute_dim):
        if model_type == "vit":
            return BirdClassifierVIT(num_classes, attribute_dim)
        elif model_type == "resnet50":
            return BirdClassifierResnet50(num_classes, attribute_dim)
        elif model_type == "eff_net_b2":
            return BirdClassifierEffNetB2(num_classes, attribute_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class BirdClassifierVIT(nn.Module):
    def __init__(self, num_classes=200, attribute_dim=312):
        super(BirdClassifierVIT, self).__init__()

        # Initialize larger ViT
        self.vit = ViTModel.from_pretrained(
            "google/vit-large-patch16-224",
            add_pooling_layer=False,  # Don't use the default pooler
            ignore_mismatched_sizes=True,
        )
        vit_output_size = 1024  # ViT large model output dimension

        # Progressive unfreezing
        self.freeze_model()

        # Multi-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(vit_output_size + attribute_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def freeze_model(self):
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x, attributes):
        vit_output = self.vit(x).last_hidden_state[:, 0, :]
        combined = torch.cat((vit_output, attributes), dim=1)
        return self.classifier(combined)


class BirdClassifierResnet50(nn.Module):
    def __init__(self, num_classes=200, attribute_dim=312):
        super(BirdClassifierResnet50, self).__init__()
        self.resnet = models.resnet50(weights="IMAGENET1K_V2")
        output_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(output_features + attribute_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, attributes):
        cnn_features = self.resnet(x)
        combined = torch.cat((cnn_features, attributes), dim=1)
        return self.classifier(combined)


class BirdClassifierEffNetB2(nn.Module):
    pass

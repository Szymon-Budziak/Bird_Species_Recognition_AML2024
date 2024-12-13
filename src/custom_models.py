import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision import models


class ModelFactory:
    @staticmethod
    def create_model(config: dict):
        model_type = config["model_type"]
        num_classes = config["num_classes"]
        attribute_dim = config["attribute_dim"]

        if model_type == "vit":
            return BirdClassifierVIT(config, num_classes, attribute_dim)
        elif model_type == "resnet":
            return BirdClassifierResnet(config, num_classes, attribute_dim)
        elif model_type == "eff_net":
            return BirdClassifierEffNet(config, num_classes, attribute_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def build_classifier(model_output_size: int, num_classes: int, attribute_dim: int):
    """
    Builds a classifier for the model.
    """
    classifier = nn.Sequential(
        nn.Linear(model_output_size + attribute_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Linear(256, num_classes),
    )
    return classifier


class BirdClassifierVIT(nn.Module):
    def __init__(self, config: dict, num_classes: int = 200, attribute_dim: int = 312):
        super(BirdClassifierVIT, self).__init__()

        vit_model = config["vit_model"]
        print(f"Using model: {vit_model}")
        vit_config = ViTConfig.from_pretrained(vit_model)

        self.backbone = ViTModel.from_pretrained(
            vit_model,
            config=vit_config,
        )

        vit_output_size = self.backbone.config.hidden_size

        # Progressive unfreezing
        n = config["unfreeze_layers"]
        print(f"Unfreezing {n} layers")
        self.freeze_model(n)

        self.classifier = build_classifier(vit_output_size, num_classes, attribute_dim)

    def freeze_model(self, n: int):
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last n layers of the backbone
        if n > 0:
            total_layers = len(self.backbone.encoder.layer)
            for layer_idx in range(total_layers - n, total_layers):
                for param in self.backbone.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True

    def forward(self, x, attributes):
        backbone_output = self.backbone(x).last_hidden_state[:, 0, :]
        combined = torch.cat((backbone_output, attributes), dim=1)
        return self.classifier(combined)


class BirdClassifierResnet(nn.Module):
    def __init__(self, config: dict, num_classes: int = 200, attribute_dim: int = 312):
        super(BirdClassifierResnet, self).__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V2")
        output_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        n = config["unfreeze_layers"]
        print(f"Unfreezing {n} layers")
        self.freeze_model(n)

        # Enhanced classifier with stronger regularization
        self.classifier = build_classifier(output_features, num_classes, attribute_dim)

    def freeze_model(self, n: int):
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last n layers of the backbone
        if n > 0:
            all_layers = list(self.backbone.named_parameters())

            for _, param in all_layers[-n:]:
                param.requires_grad = True

    def forward(self, x, attributes):
        backbone_output = self.backbone(x)
        combined = torch.cat((backbone_output, attributes), dim=1)
        return self.classifier(combined)


class BirdClassifierEffNet(nn.Module):
    def __init__(self, config: dict, num_classes: int = 200, attribute_dim: int = 312):
        super(BirdClassifierEffNet, self).__init__()
        self.backbone = models.efficientnet_b4(weights="IMAGENET1K_V1")
        output_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = build_classifier(output_features, num_classes, attribute_dim)

    def forward(self, x, attributes):
        backbone_output = self.backbone(x)
        combined = torch.cat((backbone_output, attributes), dim=1)
        return self.classifier(combined)

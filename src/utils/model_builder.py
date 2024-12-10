from torchvision import models
from transformers import ViTModel, ViTConfig
import sys

sys.path.append("..")
from custom_models import CustomEfficientNet, CustomVIT


class ModelBuilder:
    """
    Class for building models
    """
    def __init__(self):
        pass

    @staticmethod
    def build_efficientnet():
        # Load pretrained efficientnet
        pretrained_effnet = models.efficientnet_b2(weights="IMAGENET1K_V1")

        # Replace the head for transfer learning
        num_ftrs = pretrained_effnet.classifier[1].in_features

        return CustomEfficientNet(pretrained_effnet, num_ftrs)

    @staticmethod
    def build_vit():
        config = ViTConfig(
            image_size=224,
            num_classes=200,
            num_hidden_layers=12,
            hidden_size=768,
        )

        vit_model = ViTModel(config)

        return CustomVIT(vit_model)

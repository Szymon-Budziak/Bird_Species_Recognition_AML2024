import torch.nn as nn


class CustomEfficientNet(nn.Module):
    def __init__(self, backbone, num_features, config_path=None):
        super(CustomEfficientNet, self).__init__()
        # self.config = load_config(config_path)
        self.backbone = backbone
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 200),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

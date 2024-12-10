import torch.nn as nn


class CustomVIT(nn.Module):
    def __init__(self, backbone, config_path=None):
        # self.config = load_config(config_path)
        super(CustomVIT, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.2),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 200),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.backbone(x).last_hidden_state[:, 0, :]
        return self.classifier(x)

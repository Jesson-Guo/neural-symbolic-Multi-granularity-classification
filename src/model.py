import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, backbone, dim, out_planes) -> None:
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(dim, out_planes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = torch.softmax(x, dim=1)
        return x

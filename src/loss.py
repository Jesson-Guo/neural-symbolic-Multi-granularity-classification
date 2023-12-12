import torch
import torch.nn as nn
import torch.nn.functional as f


class PsychoCrossEntropy(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x, y, num_classes=None):
        x = f.normalize(x, dim=1)
        if num_classes == None:
            num_classes = self.num_classes
        x = torch.log(x+1e-5)
        y = f.one_hot(y, num_classes)
        loss = y * x
        loss = torch.sum(loss, dim=1)
        loss = -torch.mean(loss, dim=0)
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as f


class PsychoCrossEntropy(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x, y, num_classes=None, norm=False):
        if num_classes == None:
            num_classes = self.num_classes
        if norm:
            x = f.normalize(x, dim=1)
        x = torch.log(x+1e-9)
        y = f.one_hot(y, num_classes)
        loss = y * x
        loss = torch.sum(loss, dim=1)
        loss = -torch.mean(loss, dim=0)
        return loss


class PsychoClassBalancedCrossEntropy(nn.Module):
    def __init__(self, num_classes, samples_per_cls) -> None:
        super().__init__()
        self.num_classes = num_classes

        N = sum(samples_per_cls)
        beta = (N-1) / N
        effective_number = []
        for ny in samples_per_cls:
            effective_number.append((1-beta**ny) / (1-beta))
        self.effective_number = torch.FloatTensor(effective_number)

    def forward(self, x, y, effective_number=None, num_classes=None, norm=False):
        if num_classes == None:
            num_classes = self.num_classes
        if norm:
            x = f.normalize(x, dim=1)

        y = f.one_hot(y, num_classes).float()

        if effective_number == None:
            effective_number = self.effective_number.to(x.device)
        else:
            effective_number = torch.FloatTensor(effective_number).to(x.device)

        weights = 1 / effective_number
        weights = weights / torch.sum(weights) * num_classes
        weights = weights.unsqueeze(0)
        weights = weights.repeat(y.shape[0],1) * y
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, num_classes)

        loss = f.binary_cross_entropy(input=x, target=y, weight=weights)
        return loss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PsychoCrossEntropy(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x, y, samples_per_cls=None, num_classes=None, norm=False, with_logits=False):
        if num_classes == None:
            num_classes = self.num_classes

        # if norm:
        #     x = F.normalize(x, dim=1)

        # x = torch.log(x+1e-9)
        # y = F.one_hot(y, num_classes)
        # loss = y * x
        # loss = torch.sum(loss, dim=1)
        # loss = -torch.mean(loss, dim=0)
        loss = F.cross_entropy(x, y)
        return loss


class PsychoClassBalancedCrossEntropy(nn.Module):
    def __init__(self, num_classes, samples_per_cls) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.samples_per_cls = samples_per_cls

    def forward(self, x, y, samples_per_cls=None, num_classes=None, norm=False, with_logits=False):
        if num_classes == None:
            num_classes = self.num_classes
        if norm:
            x = F.normalize(x, dim=1)
        if samples_per_cls == None:
            samples_per_cls = self.samples_per_cls

        N = sum(samples_per_cls)
        beta = (N-1) / N

        effective_number = (1.0 - np.power(beta, np.array(samples_per_cls))) / (1.0 - beta)
        effective_number = torch.FloatTensor(effective_number).to(x.device)

        y = F.one_hot(y, num_classes).float()

        weights = 1 / effective_number
        weights = weights / torch.sum(weights) * num_classes

        # return F.cross_entropy(x, y, weight=weights)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(y.shape[0], 1) * y
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, num_classes)

        loss = F.binary_cross_entropy(x, y, weight=weights)
        return loss


class VSLoss(nn.Module):
    def __init__(self, num_classes, samples_per_cls, gamma=0.3, tau=1.0):
        super(VSLoss, self).__init__()
        self.num_classes = num_classes
        self.samples_per_cls = samples_per_cls
        self.gamma = gamma
        self.tau = tau

    def forward(self, x, y, samples_per_cls=None, num_classes=None, norm=False, with_logits=False):
        if num_classes == None:
            num_classes = self.num_classes
        if norm:
            x = F.normalize(x, dim=1)
        if samples_per_cls == None:
            samples_per_cls = self.samples_per_cls

        N = sum(samples_per_cls)
        beta = (N-1) / N

        samples_per_cls = np.array(samples_per_cls)
        effective_number = (1.0 - np.power(beta, samples_per_cls)) / (1.0 - beta)
        effective_number = torch.FloatTensor(effective_number).to(x.device)

        weights = 1 / effective_number
        weights = weights / torch.sum(weights) * num_classes

        cls_probs = [cls_num / sum(samples_per_cls) for cls_num in samples_per_cls]
        temp = (1.0 / samples_per_cls) ** self.gamma
        temp = temp / np.min(temp)

        iota_list = self.tau * np.log(cls_probs)
        delta_list = temp

        iota_list = torch.FloatTensor(iota_list).to(x.device)
        delta_list = torch.FloatTensor(delta_list).to(x.device)

        x = x / delta_list + iota_list

        return F.cross_entropy(x, y, weight=weights)


class LDAMLoss(nn.Module):
    def __init__(self, num_classes, samples_per_cls, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        self.num_classes = num_classes
        self.samples_per_cls = samples_per_cls
        self.max_m = max_m
        self.s = s

    def forward(self, x, y, samples_per_cls=None, num_classes=None, norm=False, with_logits=False):
        if num_classes == None:
            num_classes = self.num_classes
        if norm:
            x = F.normalize(x, dim=1)
        if samples_per_cls == None:
            samples_per_cls = self.samples_per_cls

        N = sum(samples_per_cls)
        beta = (N-1) / N

        samples_per_cls = np.array(samples_per_cls)
        effective_number = (1.0 - np.power(beta, samples_per_cls)) / (1.0 - beta)
        effective_number = torch.FloatTensor(effective_number).to(x.device)

        weights = 1 / effective_number
        weights = weights / torch.sum(weights) * num_classes

        m_list = 1.0 / np.sqrt(np.sqrt(samples_per_cls))
        m_list = m_list * (self.max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)

        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, y.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)

        return F.cross_entropy(output, y, weight=weights)

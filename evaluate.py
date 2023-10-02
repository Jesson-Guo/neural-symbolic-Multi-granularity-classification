import torch
import progressbar
import copy

from src.utils import *


@torch.no_grad()
def evaluate(dataloader, model, criterion, device):
    model.eval()
    acc = 0
    bar = progressbar.ProgressBar(0, len(dataloader))

    # for images, labels, boxes in dataloader:
    for i, (x, targets) in enumerate(dataloader):
        x = torch.autograd.Variable(x)
        x = x.to(device)
        labels = targets['labels']
        labels = labels.to(device)

        out = model(x)
        _, pred = torch.max(out.data, 1)
        # loss = criterion(out, labels)
        acc += torch.sum(pred == labels)

        bar.update(i)

    acc = acc / len(dataloader.dataset)

    return acc

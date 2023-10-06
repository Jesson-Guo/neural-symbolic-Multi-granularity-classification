import time
import progressbar

import torch

from src.utils import *
from engine.symbolic_engine import *


def train_one_epoch(dataloader, model, infer_tree, optimizer, criterion, status, device):
    model.train()
    his, epoch = status

    start = time.time()
    bar = progressbar.ProgressBar(0, len(dataloader))

    for i, (x, targets) in enumerate(dataloader):
        # label = label.cuda()
        x = torch.autograd.Variable(x)
        x = x.to(device)

        labels = torch.autograd.Variable(targets['labels'])
        labels = labels.to(device)

        out = model(x)
        # inference
        out = infer_tree.infer(out)
        loss = 0
        for i in range(len(out)):
            loss += criterion(out[0], labels[i])

        his.update(loss.item(), x.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.update(i)

    end = time.time()

    print(f'\
        Epoch: [{epoch}][{i+1}/{len(dataloader)}]\t \
        Time: {end-start}\t \
        Loss: {his.avg}\t')
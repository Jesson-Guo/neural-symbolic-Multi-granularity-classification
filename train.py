import time
import progressbar
import copy

import torch

from src.utils import *
from engine.symbolic_engine import *


def train_one_epoch(dataloader, model, infer_tree, optimizer, criterion, lpaths, status, device):
    model.train()
    his, epoch = status

    start = time.time()
    bar = progressbar.ProgressBar(0, len(dataloader))

    for i, (x, targets) in enumerate(dataloader):
        # if i > 2:
        #     break
        x = torch.autograd.Variable(x)
        x = x.to(device)

        # cifar10:
        labels = torch.autograd.Variable(targets)
        # tinyimagenet:
        # labels = torch.autograd.Variable(targets['labels'])
        labels = labels.to(device)

        out, x = model(x)
        # TODO 是否考虑原始resnet的loss ？
        # loss = criterion(out, labels)

        # inference
        loss = criterion(out, labels)
        # out, penalty = infer_tree.forward(x, targets)
        # loss += penalty

        his.update(loss.item(), x.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.update(i)
        end = time.time()
        # print(f'\
        #     Epoch: [{epoch}][{i+1}/{len(dataloader)}]\t \
        #     Time: {end-start}\t \
        #     Loss: {his.avg}\t')
        # start = end


    
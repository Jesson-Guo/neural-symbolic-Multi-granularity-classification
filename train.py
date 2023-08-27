import time
import progressbar

import torch

from src.utils import *
from engine.symbolic_engine import *


def train_one_epoch(dataloader, model, inference, optimizer, status, device):
    model.train()
    inference.train()
    his, epoch = status

    start = time.time()
    bar = progressbar.ProgressBar(0, len(dataloader))

    # for i, (images, labels, boxes) in progressbar.progressbar(enumerate(dataloader)):
    for i, (x, targets) in enumerate(dataloader):
        # label = label.cuda()
        x = torch.autograd.Variable(x)
        x.to(device)

        labels = torch.autograd.Variable(targets['labels'])
        labels.to(device)

        feat = model(x)
        feat_list = [feat['0'], feat['1'], feat['2'], feat['3']]
        loss = inference(feat_list, targets['labels'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.update(i)

    end = time.time()

    print(f'\
        Epoch: [{epoch}][{i+1}/{len(dataloader)}]\t \
        Time: {end-start}\t \
        Loss: {his.avg}\t')

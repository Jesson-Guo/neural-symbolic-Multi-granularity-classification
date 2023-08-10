import time

import torch

from src.utils import *


def train(dataloader, model, optimizer, status):
    accuracy, losses, epoch = status

    for i, (images, labels, boxes) in enumerate(dataloader):
        start = time.time()
        # label = label.cuda()
        # x = torch.autograd.Variable(x)
        # boxes = torch.autograd.Variable(target['boxes'])
        x = list(image for image in images)
        targets = []
        for i in range(boxes.shape[0]):
            targets.append({'boxes': boxes[i].reshape(1,4), 'labels': labels[i].reshape(1)})

        loss_dict = model(x, targets)
        # we only consider the box regression loss
        losses = loss_dict['loss_box_reg']

        # record best acc and loss
        # acc = compute_acc(score.data.cpu(), boxes.data.cpu(), x.data.cpu())
        # accuracy.update(acc, x.shape[0])
        # losses.update(loss.data[0], x.shape[0])

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        end = time.time()

        # if i % 10 == 0:
        #     print(f'\
        #         Epoch: [{epoch}][{i}/{len(dataloader)}]\t \
        #         Time: {end-start}\t \
        #         Loss: {loss}\t')

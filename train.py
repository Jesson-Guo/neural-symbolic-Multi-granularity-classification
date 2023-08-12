import time
import progressbar

import torch

from src.utils import *


def train(dataloader, model, optimizer, status, device):
    his, epoch = status

    start = time.time()
    bar = progressbar.ProgressBar(0, len(dataloader))

    # for i, (images, labels, boxes) in progressbar.progressbar(enumerate(dataloader)):
    for i, (images, labels, boxes) in enumerate(dataloader):
        # label = label.cuda()
        # x = torch.autograd.Variable(x)
        # boxes = torch.autograd.Variable(target['boxes'])
        x = list(image.to(device) for image in images)
        targets = []
        for j in range(boxes.shape[0]):
            targets.append({
                'boxes': boxes[j].reshape(1,4).to(device),
                'labels': labels[j].reshape(1).to(device)
            })

        loss_dict = model(x, targets)
        # we only consider the box regression loss
        losses = sum(loss for loss in loss_dict.values())
        # loss_classifier = loss_dict['loss_classifier']
        # loss_box_reg = loss_dict['loss_box_reg']
        # loss_objectness = loss_dict['loss_objectness']
        # loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']

        # loss_classifier.detach()

        # losses = loss_box_reg + loss_rpn_box_reg + loss_objectness

        # record best acc and loss
        his.update(losses.data, len(x))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        bar.update(i*len(x))

    end = time.time()

    print(f'\
        Epoch: [{epoch}][{i+1}/{len(dataloader)}]\t \
        Time: {end-start}\t \
        Loss: {his.avg}\t')

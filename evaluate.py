import torch

from src.utils import *


def evaluate(dataloader, model):
    model.eval()

    accuracy = AverageMeter()

    for images, labels, boxes in dataloader:
        # label = label.cuda()
        # x = torch.autograd.Variable(x)
        # boxes = torch.autograd.Variable(boxes)
        x = list(image for image in images)
        targets = []
        for i in range(boxes.shape[0]):
            targets.append({'boxes': boxes[i].reshape(1,4), 'labels': labels[i].reshape(1)})

        acc = model(x, targets)

        # record best acc and loss
        accuracy.update(acc, x.shape[0])

    return accuracy.avg

import torch
import progressbar

from src.utils import *


@torch.no_grad()
def evaluate(dataloader, model, device):
    loss_box_reg = AverageMeter()

    # for images, labels, boxes in dataloader:
    for _, (images, labels, boxes) in progressbar.progressbar(enumerate(dataloader)):
        # label = label.cuda()
        # x = torch.autograd.Variable(x)
        # boxes = torch.autograd.Variable(boxes)
        x = list(image.to(device) for image in images)
        targets = []
        for i in range(boxes.shape[0]):
            targets.append({
                'boxes': boxes[i].reshape(1,4).to(device),
                'labels': labels[i].reshape(1).to(device)
            })

        acc = model(x, targets)

        # record best acc and loss
        loss_box_reg.update(acc['loss_box_reg'], len(x))

    return loss_box_reg.avg

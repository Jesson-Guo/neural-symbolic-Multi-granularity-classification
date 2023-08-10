import cv2
import matplotlib.pyplot as plt

from src.utils import *


def evaluate(dataloader, model):
    loss_box_reg = AverageMeter()

    for images, labels, boxes in dataloader:
        # label = label.cuda()
        # x = torch.autograd.Variable(x)
        # boxes = torch.autograd.Variable(boxes)
        x = list(image for image in images)
        targets = []
        for i in range(boxes.shape[0]):
            targets.append({'boxes': boxes[i].reshape(1,4), 'labels': labels[i].reshape(1)})

        acc = model(x, targets)

        img = images[0].detach().numpy().transpose((1, 2, 0))
        bbox = acc[0]['boxes'].detach().numpy().reshape((4))
        x_min, y_min, x_max, y_max = bbox.astype(np.int16)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255,0,0))
        plt.imshow()
        plt.show()
        plt.savefig("./output/test")

        # record best acc and loss
        loss_box_reg.update(acc['loss_box_reg'], len(x))

    return loss_box_reg.avg

import torch
import progressbar
import copy

from src.utils import *


@torch.no_grad()
def evaluate(dataloader, model, infer_tree, lpaths, device):
    model.eval()
    acc = 0.
    # cnn_acc = 0
    bar = progressbar.ProgressBar(0, len(dataloader))

    # for images, labels, boxes in dataloader:
    for i, (x, targets) in enumerate(dataloader):
        x = torch.autograd.Variable(x)
        x = x.to(device)

        labels = torch.autograd.Variable(targets['labels'])
        labels = labels.to(device)

        out, x = model(x)
        # TODO 是否考虑原始resnet的loss ？
        # loss = criterion(out, labels)

        # inference
        out, _ = infer_tree.forward(x)
        out = torch.softmax(out, dim=1)
        pred = out.data.max(1)[1]

        acc += pred.eq(labels.data).sum()

        # for i in range(x.shape[0]):
        #     lp = copy.deepcopy(lpaths[labels[i]])
        #     lp.reverse()
        #     lp.append(labels[i])

        #     x0 = x[i].unsqueeze(0)
        #     x0 = torch.autograd.Variable(x0)
        #     x0 = x0.to(device)

        #     _, x0 = model(x0)
        #     out = infer_tree.infer(x0)

        #     for j in range(min(len(lp), len(out))):
        #         if lp[j] != out[j]:
        #             break

        #     temp += j / len(lp)
        
        # temp /= x.shape[0]
        # acc += temp

        # _, cnn_pred = torch.max(out.data, 1)
        # loss = criterion(out, labels)
        # cnn_acc += torch.sum(cnn_pred == labels)

        bar.update(i)

    acc = acc / len(dataloader.dataset)

    return acc

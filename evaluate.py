import numpy as np
import torch
import progressbar
import copy

from utils.average_meter import *
from utils.globals import *


def evaluate(dataloader, model, env, agent, infer_tree, epoch, device):
    model.eval()
    agent.eval()
    acc = 0.
    rl_acc = 0.
    # cnn_acc = 0
    bar = progressbar.ProgressBar(0, len(dataloader))

    # for images, labels, boxes in dataloader:
    for i, (x, targets) in enumerate(dataloader):
        x = torch.autograd.Variable(x)
        x = x.to(device)

        labels = torch.autograd.Variable(targets)
        labels = labels.to(device)
        targets += 1

        x = model(x)
        # TODO 是否考虑原始resnet的loss ？
        # loss = criterion(out, labels)

        # inference
        out = infer_tree.infer(x)
        pred = out.data.max(1)[1]
        acc += pred.eq(labels.data).sum()

        if epoch > 100:
            rl_pred = np.zeros(targets.shape, dtype=np.int64)
            rl_out, _, _ = agent(x, env, targets)
            for j in range(targets.shape[0]):
                rl_pred[j] = get_value('label2id')[rl_out[j][-1]] - 1
            rl_pred = torch.as_tensor(rl_pred).to(device)
            rl_acc += rl_pred.eq(labels.data).sum()

        bar.update(i+1)

    return acc, rl_acc

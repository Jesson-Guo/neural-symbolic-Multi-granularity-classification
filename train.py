import time
import progressbar
import copy

import torch

from utils.average_meter import *


def train_one_epoch(dataloader, model, infer_tree, env, agent, optimizer, criterion, status, device):
    model.train()
    agent.train()
    his, epoch = status

    start = time.time()
    # bar = progressbar.ProgressBar(0, len(dataloader))

    for i, (x, targets) in enumerate(dataloader):
        # if i > 1:
        #     break
        x = torch.autograd.Variable(x)
        x = x.to(device)

        labels = torch.autograd.Variable(targets)
        labels = labels.to(device)
        targets += 1

        # out = torch.autograd.Variable(out)
        out, x = model(x)
        # TODO 是否考虑原始resnet的loss ？
        # loss = criterion(out, labels)

        # inference
        out, loss = infer_tree.forward(x, targets)
        # loss += infer_tree.cross_entrophy(out, labels, mode='prob')
        loss += criterion(out, labels)

        total_reward = 0
        _, total_reward, dqn_loss = agent.run_batch(x, env, targets)

        his.update(loss.item(), x.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # bar.update(i)

        end = time.time()

        print(f'\
            Epoch: [{epoch}][{i+1}/{len(dataloader)}]\t \
            Time: {end-start}\t \
            Loss: {his.avg}\t \
            Reward: {total_reward} \
            DQN Loss: {dqn_loss}')
        start = end

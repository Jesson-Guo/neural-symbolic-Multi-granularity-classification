import time

import torch


def train(dataloader, model, criterion, optimizer, status):
    accuracy, losses, epoch = status

    # switch to train mode
    model.train()

    for i, (x, label) in enumerate(dataloader):
        start = time.time()
        # label = label.cuda()
        x = torch.autograd.Variable(x)
        label = torch.autograd.Variable(label)

        score = model(x)
        loss = criterion(score, label)

        # record best acc and loss
        _, pred = score.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))
        prec1 = correct[:1].view(-1).float().sum(0, keepdim=True)
        prec1 = prec1.mul_(100.0 / label.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % 10 == 0:
            print(f'\
                Epoch: [{epoch}][{i}/{len(dataloader)}]\t \
                Time: {end-start}\t \
                Loss: {loss}\t \
                prec@1: {prec1}\t')

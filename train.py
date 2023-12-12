import time
import tqdm

import torch
import torch.distributed

from utils.util import reduce_mean, AverageMeter, accuracy
from utils.conf import is_main_process


def train_one_epoch(dataloader, model, infer_tree, optimizer, criterion, epoch, device):
    model.train()
    train_loss = AverageMeter()
    acc = torch.zeros(2).to(device)
    data_len = len(dataloader.dataset)

    start = time.time()

    if is_main_process():
        dataloader = tqdm.tqdm(dataloader)

    for i, (x, targets) in enumerate(dataloader):
        x = torch.autograd.Variable(x)
        x = x.to(device)

        labels = torch.autograd.Variable(targets)
        labels = labels.to(device)
        targets += 1

        x = model(x)
        # loss = criterion(out, labels)

        # inference
        out = infer_tree(x, targets)
        loss = criterion(out, labels)

        acc1, acc2 = accuracy(out, labels, topk=(1, 2))
        acc[0] += acc1
        acc[1] += acc2

        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, average=True)
        train_loss.update(reduced_loss.item(), x.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        if is_main_process():
            dataloader.desc = f'\
                Epoch: [{epoch+1}][{i+1}/{len(dataloader)}]\t \
                Time: {end-start}\t \
                Loss: {reduced_loss.item()}\t'
        start = end

    torch.distributed.barrier()
    acc = reduce_mean(acc, average=False)
    acc = acc / data_len
    if is_main_process():
        print(f'\
            train top1: {acc[0].item()}\t \
            train top2: {acc[1].item()}')
    return train_loss.avg, acc

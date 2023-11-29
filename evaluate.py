import time
import torch
import torch.distributed
import progressbar

from utils.util import *
from utils.globals import *
from utils.conf import is_main_process


def evaluate(dataloader, model, infer_tree, criterion, epoch, device):
    model.eval()
    eval_loss = AverageMeter()
    acc = torch.zeros(2).to(device)

    if is_main_process():
        bar = progressbar.ProgressBar(0, len(dataloader))
    start = time.time()

    with torch.no_grad():
        for i, (x, targets) in enumerate(dataloader):
            x = x.to(device)
            labels = targets.to(device)
            targets += 1

            x = model(x)
            out, loss = infer_tree(x, targets)
            loss += criterion(out, labels)

            acc1, acc2 = accuracy(out, labels, topk=(1, 2))
            acc[0] += acc1
            acc[1] += acc2

            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss)
            eval_loss.update(reduced_loss.item(), x.shape[0])

            if is_main_process():
                bar.update(i+1)

    torch.distributed.barrier()
    end = time.time()
    acc = reduce_mean(acc, average=False)
    acc = acc / len(dataloader.dataset)
    if is_main_process():
        print(f'\
            Epoch: [{epoch+1}]\t \
            eval top1: {acc[0].item()}\t \
            eval top2: {acc[1].item()}\t \
            time: {end - start}')

    return eval_loss.avg, acc

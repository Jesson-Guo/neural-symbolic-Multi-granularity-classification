import progressbar
import torch
import torch.utils.data

from utils.conf import is_main_process
from utils.util import accuracy, reduce_mean


@torch.no_grad()
def eval(cfg, tot, model, val_loader, alpha, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_len = len(val_loader.dataset)
    acc = torch.zeros(2).to(device)

    if is_main_process():
        bar = progressbar.ProgressBar(0, len(val_loader))

    for idx, (x, targets) in enumerate(val_loader):
        x = x.to(device)
        targets = targets.to(device)

        if cfg.METHOD == "tot":
            tot.clean()
            x, corase_x = model(x, return_feature=True)
            x = torch.cat([x, corase_x], dim=1)
            pred = tot.solve(x, alpha, method='dfs')
            acc[0] += pred.eq(targets.data).sum()
        elif cfg.METHOD == "vit":
            outputs = model(x)
            acc1, acc2 = accuracy(outputs, targets, topk=(1, 5))
            acc[0] += acc1
            acc[1] += acc2

        if cfg.NUM_GPUS > 1:
            torch.distributed.barrier()

        if is_main_process():
            bar.update(idx+1)

    acc = reduce_mean(acc, average=False)
    acc = acc / data_len
    if is_main_process():
        print(f'\
            train top1: {acc[0].item()}\t\
            train top5: {acc[1].item()}')

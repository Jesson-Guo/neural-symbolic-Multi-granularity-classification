import progressbar
import torch
import torch.utils.data

from utils.conf import is_main_process
from utils.util import accuracy, reduce_mean


@torch.no_grad()
def eval(cfg, tot, model, val_loader, num_classes, alpha, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_len = len(val_loader.dataset)
    classes = val_loader.dataset.classes
    img_num_list = val_loader.dataset.img_num_list

    acc = torch.zeros(2).to(device)
    wrong_acc = torch.zeros(num_classes).to(device)

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
            pred = outputs.max(1)[1]

        wrong_indices = torch.nonzero(pred.data!=targets.data)
        wrong_targets = targets[wrong_indices]
        for i in range(num_classes):
            wrong_acc[i] += (wrong_targets.data==i).sum()

        if cfg.NUM_GPUS > 1:
            torch.distributed.barrier()

        if is_main_process():
            bar.update(idx+1)

    wrong_acc = reduce_mean(wrong_acc, average=False)
    acc = reduce_mean(acc, average=False)
    acc = acc / data_len
    if is_main_process():
        print(f'\
            val top1: {acc[0].item()}\t\
            val top5: {acc[1].item()}')
        for i in range(num_classes):
            print(f"{classes[i]}: ({wrong_acc[i].item()}, {wrong_acc[i].item() / img_num_list[i]})")
    return acc

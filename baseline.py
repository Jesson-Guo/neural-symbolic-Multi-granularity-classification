import argparse
import os
import random
import progressbar

import timm
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from src.dataloader import create_train_val_dataloader
from utils.conf import get_world_size, is_main_process
from utils.util import accuracy, reduce_mean


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    model = timm.create_model(
        model_name=args.model,
        pretrained=True,
        pretrained_cfg_overlay=dict(file=args.ckpt),
    ).to(device)

    if get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # find_unused_parameters=True
        )

    _, val_loader = create_train_val_dataloader(args)

    model.eval()
    acc = torch.FloatTensor([0.]).to(device)
    if is_main_process():
        bar = progressbar.ProgressBar(0, len(val_loader))
    with torch.no_grad():
        for i, (x, targets) in enumerate(val_loader):
            x = x.to(device)
            targets = targets.to(device)

            output = model(x)
            output = output.softmax(dim=1)
            # top1, top5 = accuracy(output, targets, topk=(1, 5))
            # acc += top1
            pred = output.data.max(1)[1]
            acc += pred.eq(targets.data).sum()

            if is_main_process():
                bar.update(i+1)

    acc = reduce_mean(acc, average=False)
    acc = acc / len(val_loader.dataset)
    if is_main_process():
        print(f'eval acc: {acc.item()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tiny imagenet training')

    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224.orig_in21k_ft_in1k', help='model name')
    parser.add_argument('--ckpt', type=str, default='./mycode/weights/jx_vit_base_p16_224-80ecf9dd.pth', help='path of model checkpoint')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data', help='dataset path')
    parser.add_argument('--data', type=str, default='imagenet', help='dataset name')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    args = parser.parse_args()
    main(args)

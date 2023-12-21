import argparse
import os
import random
import openai

import timm
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from functools import partial

from src.dataloader import create_train_val_dataloader
from src.node import build_tree
from src.gpt import GPT
from src.tot.infer import solve
from src.tot.tot import ToT
from utils import metrics
from utils.conf import get_world_size


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

    _, val_loader = create_train_val_dataloader(args)

    state_dict = model.state_dict()
    tree, node_dict = build_tree(args, val_loader.dataset.class_to_idx, state_dict[''])

    client = openai.OpenAI()
    gpt = GPT(client, model=args.backend, temperature=args.temperature)

    sim_func = getattr(metrics, args.sim)
    plan_func = getattr(metrics, args.plan)
    tot = ToT(plan_func, sim_func)

    if get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # find_unused_parameters=True
        )

    solve(model, val_loader, tree, node_dict, gpt, tot)


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
    parser.add_argument('--backend', type=str, default='gpt-3.5-turbo', help='gpt model')
    parser.add_argument('--sim', type=str, default='kl_divergence', help='similarity metrics')
    parser.add_argument('--plan', type=str, default='kl_divergence', help='cluster metrics')

    args = parser.parse_args()
    main(args)

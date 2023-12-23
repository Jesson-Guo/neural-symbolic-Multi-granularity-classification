import argparse
import os
import random
import openai

import timm
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from src.dataloader import create_train_val_dataloader
from src.node import build_tree
from src.gpt import FakeGPT
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

    model = timm.create_model(model_name=args.model, pretrained=False).to(device)

    _, val_loader = create_train_val_dataloader(args)

    state_dict = model.state_dict()
    tree, node_dict, label_to_wnid, label_to_id, labels = build_tree(args, val_loader.dataset.class_to_idx, state_dict['head.weight'])

    client = openai.OpenAI()
    gpt = FakeGPT(client, model=args.backend)

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

    solve(model, val_loader, node_dict, label_to_wnid, label_to_id, labels, device, gpt, tot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification with gpt')

    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224.orig_in21k_ft_in1k', help='model name')
    parser.add_argument('--hier', type=str, default='./structure_released.xml', help='wordnet structure')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data/test', help='dataset path')
    parser.add_argument('--data', type=str, default='imagenet', help='dataset name')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--backend', type=str, default='gpt-3.5-turbo', help='gpt model')
    parser.add_argument('--sim', type=str, default='kl_divergence', help='similarity metrics')
    parser.add_argument('--plan', type=str, default='silhouette_score', help='cluster metrics')

    args = parser.parse_args()
    main(args)

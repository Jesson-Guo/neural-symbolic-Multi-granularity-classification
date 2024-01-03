import argparse
import os
import random
import openai

import timm
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import metrics
from src.dataloader import create_val_dataloader
from src.node import build_tree
from src.tot.infer import solve
from src.tot.tot import ToT
from utils.conf import get_world_size
from utils.util import get_coarse_labels


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

    # model = timm.create_model(
    #     model_name=args.model,
    #     pretrained=args.pretrained,
    #     pretrained_cfg_overlay=dict(file=args.ckpt),
    #     num_classes=args.classes
    # ).to(device)
    model = timm.create_model(args.model, pretrained=False)
    model.head = nn.Linear(model.head.in_features, args.classes)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(device)

    val_loader = create_val_dataloader(args)

    state_dict = model.state_dict()
    node_dict, label_to_wnid, label_to_id, labels, _ = build_tree(args, val_loader.dataset.class_to_idx, state_dict['head.weight'])

    sim_func = getattr(metrics, args.sim)
    plan_func = getattr(metrics, args.plan)
    tot = ToT(plan_func, sim_func)
    tot.load(args.load, labels)

    if get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # find_unused_parameters=True
        )

    # solve(model, val_loader, node_dict, label_to_wnid, label_to_id, device, tot, 10)

    coarse = get_coarse_labels(tot.root)
    for i in range(60, 140):
        solve(model, val_loader, node_dict, label_to_wnid, label_to_id, device, tot, i/10)
        print(i/10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification with gpt')

    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--model', type=str, default='timm model name', help='model name')
    parser.add_argument('--pretrained', action= "store_true", help = "")
    parser.add_argument('--hier', type=str, default='./structure_released.xml', help='wordnet structure')
    parser.add_argument('--ckpt', type=str, default='/path/to/checkpoint', help='path of model checkpoint')
    parser.add_argument('--root', type=str, default='/path/to/dataset', help='dataset path')
    parser.add_argument('--data', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--classes', type=int, default=1000, help='number of classes')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--backend', type=str, default='gpt-4-1106-preview', help='gpt model')
    parser.add_argument('--temperature', type=float, default=0.7, help='gpt model temperature')
    parser.add_argument('--sim', type=str, default='naive_score', help='similarity metrics')
    parser.add_argument('--plan', type=str, default='silhouette_score', help='cluster metrics')
    parser.add_argument('--save', type=str, default='/path/to/save', help='thought file path')
    parser.add_argument('--load', type=str, default='', help='thought file path')
    parser.add_argument('--words', type=str, default='/path/to/words', help='words file path')

    args = parser.parse_args()
    main(args)

import argparse
import os
import random
import openai

import timm
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from src.dataloader import create_val_dataloader
from src.node import build_tree
from src.gpt import GPT
from src.tot.infer import solve
from src.tot.tot import ToT
from utils import metrics
from utils.conf import get_world_size


def main(args):
    model = timm.create_model(
        model_name=args.model,
        pretrained=args.pretrained,
        pretrained_cfg_overlay=dict(file=args.ckpt),
        num_classes=args.classes
    )

    val_loader = create_val_dataloader(args)

    state_dict = model.state_dict()
    node_dict, label_to_wnid, _, labels, node_children = build_tree(args, val_loader.dataset.class_to_idx, state_dict['head.weight'])

    client = openai.OpenAI()
    gpt = GPT(client, model=args.backend, temperature=args.temperature)

    sim_func = getattr(metrics, args.sim)
    plan_func = getattr(metrics, args.plan)
    tot = ToT(plan_func, sim_func)
    if args.load:
        tot.load(args.load, labels)
    tot.build_tot(labels, node_dict, label_to_wnid, node_children, node_dict['fall11'], gpt, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification with gpt')

    parser.add_argument('--model', type=str, default='timm model name', help='model name')
    parser.add_argument('--pretrained', action= "store_true", help = "")
    parser.add_argument('--hier', type=str, default='./structure_released.xml', help='wordnet structure')
    parser.add_argument('--ckpt', type=str, default='/path/to/checkpoint', help='path of model checkpoint')
    parser.add_argument('--root', type=str, default='/path/to/dataset', help='dataset path')
    parser.add_argument('--data', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--classes', type=int, default=10, help='number of classes')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--backend', type=str, default='gpt-4-1106-preview', help='gpt model')
    parser.add_argument('--temperature', type=float, default=0.7, help='gpt model temperature')
    parser.add_argument('--sim', type=str, default='kl_divergence', help='similarity metrics')
    parser.add_argument('--plan', type=str, default='silhouette_score', help='cluster metrics')
    parser.add_argument('--save', type=str, default='/path/to/save', help='thought file path')
    parser.add_argument('--load', type=str, default='', help='thought file path')
    parser.add_argument('--words', type=str, default='/path/to/words', help='words file path')

    args = parser.parse_args()
    main(args)

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
from src.tot.builder import ToTBuilder


def main(args):
    val_loader = create_val_dataloader(args)

    node_dict, label_to_wnid, _, labels, node_children = build_tree(args, val_loader.dataset.class_to_idx)

    sim_func = getattr(metrics, args.sim)
    plan_func = getattr(metrics, args.plan)
    tot = ToT(plan_func, sim_func)
    tot.load('./cifar10.json', labels)

    tot.save('./cifar10_t.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification with gpt')

    parser.add_argument('--model', type=str, default='timm model name', help='model name')
    parser.add_argument('--pretrained', action= "store_true", help = "")
    parser.add_argument('--hier', type=str, default='./structure_released.xml', help='wordnet structure')
    parser.add_argument('--ckpt', type=str, default='/path/to/checkpoint', help='path of model checkpoint')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp/data', help='dataset path')
    parser.add_argument('--data', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--classes', type=int, default=1000, help='number of classes')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--backend', type=str, default='gpt-4-1106-preview', help='gpt model')
    parser.add_argument('--temperature', type=float, default=0.8, help='gpt model temperature')
    parser.add_argument('--sim', type=str, default='kl_divergence', help='similarity metrics')
    parser.add_argument('--plan', type=str, default='silhouette_score', help='cluster metrics')
    parser.add_argument('--save', type=str, default='/path/to/save', help='thought file path')
    parser.add_argument('--load', type=str, default='', help='thought file path')
    parser.add_argument('--words', type=str, default='/path/to/words', help='words file path')

    args = parser.parse_args()
    main(args)

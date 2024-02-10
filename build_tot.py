import argparse
import openai

import timm
import torch
import torch.nn as nn
import torch.utils.data

from src.dataloader import create_val_dataloader
from src.node import build_tree
from src.gpt import GPT
from src.tot.tot import ToT
from src.tot.builder import ToTBuilder
from src.vpt.configs.config import get_cfg
from utils import metrics


def main(args):
    cfg = get_cfg()
    cfg.merge_from_file("./src/vpt/configs/files/simple/cifar100.yaml")
    cfg.WORKERS = args.workers

    val_loader = create_val_dataloader(cfg)

    node_dict, label_to_wnid, label_to_id, labels, node_children = build_tree(args, val_loader.dataset.class_to_idx)

    client = openai.OpenAI()
    gpt = GPT(client, model=args.backend, temperature=args.temperature)

    plan_func = getattr(metrics, args.plan)

    builder = ToTBuilder(plan_func, num_plans=2, num_coarse=10000, num_k=3, max_depth=2)
    root = builder.build_on_gpt(labels, gpt, "./tots/cifar100-1.json", "./tots/cifar100-1.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification with gpt')

    parser.add_argument('--hier', type=str, default='./structure_released.xml', help='wordnet structure')
    parser.add_argument('--root', type=str, default='/data/zanghan/ghj/data', help='dataset path')
    parser.add_argument('--data', type=str, default='cifar100', help='dataset name')
    parser.add_argument('--classes', type=int, default=100, help='number of classes')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--backend', type=str, default='gpt-4-1106-preview', help='gpt model')
    parser.add_argument('--temperature', type=float, default=0.8, help='gpt model temperature')
    parser.add_argument('--plan', type=str, default='silhouette_score', help='cluster metrics')
    parser.add_argument('--save', type=str, default='/path/to/save', help='thought file path')
    parser.add_argument('--load', type=str, default='', help='thought file path')
    parser.add_argument('--words', type=str, default='/data/zanghan/ghj/data/cifar-100-python/words.txt', help='words file path')

    args = parser.parse_args()
    main(args)

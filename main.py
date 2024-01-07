import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import metrics
from src.dataloader import create_val_dataloader, create_train_dataloader
from src.node import build_tree
from src.tot.tot import ToT
from src.tot.builder import ToTBuilder
from src.vpt.models.vit_models import ViT
from src.vpt.configs.config import get_cfg
from src.solver.loss import PsychoCrossEntropy
from src.solver.lr_scheduler import make_scheduler
from src.solver.optimizer import make_optimizer
from utils.conf import is_main_process
from train import train
from eval import eval


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    cudnn.benchmark = True

    cfg = get_cfg()
    cfg.merge_from_file(f"./src/vpt/configs/files/prompt/{args.data}.yaml")
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.BASE_LR = args.lr / 256 * cfg.DATA.BATCH_SIZE
    cfg.METHOD = args.method
    cfg.DATA.NUMBER_COARSE = 0

    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # 图片统一 224*224（考虑一下32*32）
    train_loader = create_train_dataloader(args)
    val_loader = create_val_dataloader(args)

    tot = None

    if cfg.METHOD == "tot":
        _, _, _, labels, _ = build_tree(args, val_loader.dataset.class_to_idx)

        sim_func = getattr(metrics, args.sim)
        plan_func = getattr(metrics, args.plan)
        builder = ToTBuilder(plan_func, num_plans=2, num_coarse=10000, num_k=5)
        root, plan_dict = builder.load(f"./{cfg.DATA.NAME}-{args.k}.json", labels)
        tot = ToT(sim_func, plan_dict, root)
        tot.reset()

        # 这里可以考虑一下是否固定叶子的weight，直接用预训练的参数还是重新训练
        # 不固定，重新训练
        cfg.DATA.NUMBER_COARSE = tot.num_others

    model = ViT(cfg)
    model = model.to(device)

    optimizer = make_optimizer([model], cfg.SOLVER)
    scheduler = make_scheduler(optimizer, cfg.SOLVER)

    criterion = PsychoCrossEntropy(cfg.DATA.NUMBER_CLASSES)

    if cfg.NUM_GPUS > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # find_unused_parameters=True
        )

    if is_main_process():
        print(cfg)

    if args.train:
        train(cfg, tot, model, criterion, optimizer, scheduler, train_loader, cfg.DATA.NUMBER_CLASSES, args.epochs, device)
        print("training over")
    elif args.test:
        eval(cfg, tot, model, val_loader, device)
        print("testing over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification with gpt')

    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--hier', type=str, default='./structure_released.xml', help='wordnet structure')
    parser.add_argument('--root', type=str, default='/path/to/dataset', help='dataset path')
    parser.add_argument('--data', type=str, default='cifar100', help='dataset name')
    parser.add_argument('--method', type=str, default='vpt', help='dataset name')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--backend', type=str, default='gpt-4-1106-preview', help='gpt model')
    parser.add_argument('--temperature', type=float, default=0.7, help='gpt model temperature')
    parser.add_argument('--sim', type=str, default='naive_score', help='similarity metrics')
    parser.add_argument('--plan', type=str, default='silhouette_score', help='cluster metrics')
    parser.add_argument('--save', type=str, default='/path/to/save', help='thought file path')
    parser.add_argument('--k', type=int, default=5, help='number k')
    parser.add_argument('--words', type=str, default='/path/to/words', help='words file path')
    parser.add_argument('--train', action= "store_true", help = "")
    parser.add_argument('--test', action= "store_true", help = "")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    main(args)

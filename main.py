import argparse
import os
import random
import traceback
import time
import tqdm
from iopath.common.file_io import HTTPURLHandler, PathManager
from typing import Any, cast, Dict, IO

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import metrics
from src.dataloader import create_val_dataloader, create_train_dataloader
from src.node import build_tree, init_weight
from src.tot.infer import solve
from src.tot.tot import ToT
from src.tot.builder import ToTBuilder
from src.vpt.models.vit_models import ViT
from src.vpt.configs.config import get_cfg
from src.solver.loss import PsychoCrossEntropy
from src.solver.lr_scheduler import make_scheduler
from src.solver.optimizer import make_optimizer
from utils.conf import is_main_process
from utils.util import AverageMeter, accuracy, reduce_mean


def eval(model, val_loader, node_dict, label_to_wnid, label_to_id, device, tot):
    init_weight(node_dict['fall11'], 0)
    solve(model, val_loader, node_dict, label_to_wnid, label_to_id, device, tot, 8)


def compute_penalty(x, targets, criterion, tot):
    penalty = 0.
    score_dict = {}

    for k, _ in tot.plan_dict.items():
        plans = []
        do_loss = []
        for ts in _.values():
            if ts[0].is_valid():
                plan_w = []
                for i in range(len(ts)):
                    plan_w.append(x[:, ts[i].tid])

                coarse_targets = torch.ones_like(targets) * (len(ts)-1)
                # if ts[len(ts)-1].name == "Other":
                #     do_loss.append(True)
                #     for i in range(targets.shape[0]):
                #         for j in range(len(ts)):
                #             if targets[i].item() in ts[j].labels:
                #                 coarse_targets[i] = j
                #                 break
                # else:
                #     do_loss.append(False)

                plans.append((plan_w, ts, coarse_targets))

        score_dict[k] = []
        for i in range(len(plans)):
            plan_w, ts, coarse_targets = plans[i]
            coarse_x = torch.stack(plan_w).T
            coarse_out = coarse_x.softmax(dim=1)
            # if do_loss[i]:
            #     penalty += criterion(coarse_out, coarse_targets, num_classes=len(ts))
            score_dict[k].append(coarse_out)
    return penalty, score_dict


def train_one_batch(x, targets, criterion, tot, num_classes, device):
    penalty, score_dict = compute_penalty(x, targets, criterion, tot)

    outputs = torch.zeros([x.shape[0], num_classes], dtype=torch.float32).to(device)
    tot.root.score = torch.FloatTensor([1]).to(device)
    tot.root.path_score = torch.FloatTensor([1]).to(device)

    thoughts = [tot.root]
    while len(thoughts):
        t = thoughts.pop()
        if t.stop():
            label_id = list(t.labels.keys())[0]
            outputs[:, label_id] += t.path_score

        label_list = list(t.labels.keys())
        label_list.sort()
        label_str = str(label_list)[1:-1]
        for i, ts in t.plans.items():
            for j in range(len(ts)):
                score = score_dict[label_str][i][:, j]
                ts[j].path_score = score * t.path_score
                thoughts.append(ts[j])

    return outputs, penalty


def train(cfg, tot, model, criterion, optimizer, scheduler, train_loader, num_classes, total_epoch, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    losses = AverageMeter()
    batch_time = AverageMeter()

    data_len = len(train_loader.dataset)
    path_manager = PathManager()
    path_manager.register_handler(HTTPURLHandler())
    save_file = os.path.join(cfg.OUTPUT_DIR, f"{cfg.METHOD}_{cfg.DATA.NAME}-10.pth")
    acc = torch.zeros(2).to(device)

    if cfg.METHOD == "vpt":
        criterion = nn.CrossEntropyLoss()

    for epoch in range(total_epoch):
        losses.reset()
        batch_time.reset()

        end = time.time()
        if is_main_process():
            train_loader = tqdm.tqdm(train_loader)

        for idx, (x, targets) in enumerate(train_loader):
            x = x.to(device)
            targets = targets.to(device)

            if cfg.METHOD == "tot":
                tot.clean()
                x = model(x)
                outputs, penalty = train_one_batch(x, targets, criterion, tot, num_classes, device)
                loss = criterion(outputs, targets, norm=True)
                loss += penalty
            elif cfg.METHOD == "vpt":
                outputs = model(x)
                loss = criterion(outputs, targets)

            acc1, acc2 = accuracy(outputs, targets, topk=(1, 5))
            acc[0] += acc1
            acc[1] += acc2

            if cfg.NUM_GPUS > 1:
                torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, average=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(reduced_loss.item(), x.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            if is_main_process():
                train_loader.desc = f"\
                    epoch: [{epoch+1}/{total_epoch}]\t\
                    batch: [{idx+1}/{len(train_loader)}]\t\
                    average train loss: {losses.avg}"

        scheduler.step()

        acc = reduce_mean(acc, average=False)
        acc = acc / data_len
        if is_main_process():
            print(f'\
                train top1: {acc[0].item()}\t\
                train top5: {acc[1].item()}')

            if cfg.NUM_GPUS > 1:
                data = {"model": model.module.state_dict()}
            else:
                data = {"model": model.state_dict()}

            with path_manager.open(save_file, "wb") as f:
                torch.save(data, cast(IO[bytes], f))


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

    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # 图片统一 224*224（考虑一下32*32）
    train_loader = create_train_dataloader(args)
    val_loader = create_val_dataloader(args)

    _, _, _, labels, _ = build_tree(args, val_loader.dataset.class_to_idx)

    sim_func = getattr(metrics, args.sim)
    plan_func = getattr(metrics, args.plan)
    builder = ToTBuilder(plan_func, num_plans=2, num_coarse=10000, num_k=5)
    root, plan_dict = builder.load(f"./{args.data}-{args.k}.json", labels)
    tot = ToT(sim_func, plan_dict, cfg.DATA.NUMBER_CLASSES, root)
    tot.reset()

    # 这里可以考虑一下是否固定叶子的weight，直接用预训练的参数还是重新训练
    # 不固定，重新训练
    num_classes = cfg.DATA.NUMBER_CLASSES
    cfg.DATA.NUMBER_CLASSES = tot.num_others

    model = ViT(cfg)
    model = model.to(device)

    optimizer = make_optimizer([model], cfg.SOLVER)
    scheduler = make_scheduler(optimizer, cfg.SOLVER)

    criterion = PsychoCrossEntropy(num_classes)

    if cfg.NUM_GPUS > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # find_unused_parameters=True
        )

    if is_main_process():
        print(cfg)
        print("training now...")

    train(cfg, tot, model, criterion, optimizer, scheduler, train_loader, num_classes, args.epochs, device)
    print("training over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification with gpt')

    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--hier', type=str, default='./structure_released.xml', help='wordnet structure')
    parser.add_argument('--root', type=str, default='/path/to/dataset', help='dataset path')
    parser.add_argument('--data', type=str, default='cifar100', help='dataset name')
    parser.add_argument('--method', type=str, default='tot', help='dataset name')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--backend', type=str, default='gpt-4-1106-preview', help='gpt model')
    parser.add_argument('--temperature', type=float, default=0.7, help='gpt model temperature')
    parser.add_argument('--sim', type=str, default='naive_score', help='similarity metrics')
    parser.add_argument('--plan', type=str, default='silhouette_score', help='cluster metrics')
    parser.add_argument('--save', type=str, default='/path/to/save', help='thought file path')
    parser.add_argument('--k', type=int, default=5, help='number k')
    parser.add_argument('--words', type=str, default='/path/to/words', help='words file path')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    main(args)

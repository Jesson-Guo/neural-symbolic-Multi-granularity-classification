import argparse
import os
import random
import traceback
import time
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
from utils.conf import get_world_size
from utils.util import AverageMeter, accuracy


def get_coarse_num(root, num_classes):
    num_coarse = num_classes-1
    leaf_to_coarse = {}
    for i in range(num_classes):
        leaf_to_coarse[i] = set()

    ts = [root]
    while len(ts):
        t = ts.pop()
        if t.stop():
            t.tid = list(t.labels.keys())[0]
            continue

        t.tid = num_coarse
        num_coarse += 1
        for _ in t.plans.values():
            for child in _:
                ts.insert(0, child)

    return num_coarse-num_classes, leaf_to_coarse


def eval(model, val_loader, node_dict, label_to_wnid, label_to_id, device, tot):
    init_weight(node_dict['fall11'], 0)
    solve(model, val_loader, node_dict, label_to_wnid, label_to_id, device, tot, 8)


def train_one_batch(x, targets, model, criterion, thought, num_classes, device):
    outputs = torch.zeros([x.shape[0], num_classes], dtype=torch.float32).to(device)
    penalty = 0.

    x, _ = model(x, return_feature=True)

    thoughts = [thought]
    while len(thoughts):
        t = thoughts.pop()
        if t.stop():
            label_id = list(t.labels.keys())[0]
            outputs[:, label_id] += t.score

        plans = []
        for ts in t.plans.values():
            if ts[0].is_valid():
                plan_w = []
                for i in range(len(ts)):
                    plan_w.append(model.head.last_layer.weight[ts[i].tid])

                coarse_targets = torch.ones_like(targets) * (len(ts)-1)
                for i in range(targets.shape[0]):
                    for j in range(len(ts)-1):
                        if len(ts[j].labels) and targets[i].item() in ts[j].labels:
                            coarse_targets[i] = j
                            break

                plans.append((plan_w, ts, coarse_targets))

        for plan_w, ts, coarse_targets in plans:
            # choose a plan and calculate score
            out = torch.zeros([x.shape[0], len(plan_w)], dtype=torch.float32).to(device)
            for i in range(len(plan_w)):
                y = plan_w[i].unsqueeze(0)
                out[:, i] = torch.matmul(x, y.T).squeeze()
            out = out.softmax(dim=1)
            try:
                penalty += criterion(out, coarse_targets, num_classes=len(ts))
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print(t.name)
                exit(0)
            for j in range(len(ts)):
                ts[j].score = out[:, j] * t.score
                thoughts.append(ts[j])
    return outputs, penalty


def train(tot, model, criterion, optimizer, scheduler, train_loader, num_classes, total_epoch, device, mode="tot"):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    losses = AverageMeter()
    batch_time = AverageMeter()
    acc = torch.zeros(2).to(device)

    if mode == "vpt":
        criterion = nn.CrossEntropyLoss()

    for epoch in range(total_epoch):
        losses.reset()
        batch_time.reset()

        end = time.time()

        for idx, (x, targets) in enumerate(train_loader):
            x = x.to(device)
            targets = targets.to(device)

            if mode == "tot":
                tot.clean()
                outputs, penalty = train_one_batch(x, targets, model, criterion, tot.root, num_classes, device)
                loss = criterion(outputs, targets, norm=True)
                loss += penalty
            elif mode == "vpt":
                outputs = model(x)
                loss = criterion(outputs, targets)

            acc1, acc2 = accuracy(outputs, targets, topk=(1, 5))
            acc[0] += acc1
            acc[1] += acc2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), x.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            print(f"epoch: [{epoch+1}/{total_epoch}]\tbatch: [{idx}]\taverage train loss: {losses.avg}")

        scheduler.step()

        acc = acc / len(train_loader.dataset)
        print(f'\
            train top1: {acc[0].item()}\t\
            train top5: {acc[1].item()}')


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    cudnn.benchmark = True

    cfg = get_cfg()
    cfg.merge_from_file(f"./src/vpt/configs/files/prompt/{args.data}.yaml")
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.BASE_LR = args.lr / 256 * cfg.DATA.BATCH_SIZE

    print(cfg)

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
    builder = ToTBuilder(2, plan_func)
    tot = ToT(sim_func)
    tot.root = builder.load(args.load, labels)

    # 这里可以考虑一下是否固定叶子的weight，直接用预训练的参数还是重新训练
    # 这是不固定，重新训练
    num_coarses, leaf_to_coarse = get_coarse_num(tot.root, cfg.DATA.NUMBER_CLASSES)
    if args.method == "tot":
        cfg.DATA.NUMBER_CLASSES += num_coarses

    model = ViT(cfg)
    model = model.to(device)

    optimizer = make_optimizer([model], cfg.SOLVER)
    scheduler = make_scheduler(optimizer, cfg.SOLVER)

    criterion = PsychoCrossEntropy(cfg.DATA.NUMBER_CLASSES)

    if get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # find_unused_parameters=True
        )

    train(tot, model, criterion, optimizer, scheduler, train_loader, cfg.DATA.NUMBER_CLASSES, args.epochs, device, mode=args.method)

    path_manager = PathManager()
    path_manager.register_handler(HTTPURLHandler())
    save_file = os.path.join(cfg.OUTPUT_DIR, f"{args.method}_{args.data}.pth")
    data = {"model": model.state_dict()}
    with path_manager.open(save_file, "wb") as f:
        torch.save(data, cast(IO[bytes], f))
    print("training over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification with gpt')

    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--hier', type=str, default='./structure_released.xml', help='wordnet structure')
    parser.add_argument('--root', type=str, default='/path/to/dataset', help='dataset path')
    parser.add_argument('--data', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--method', type=str, default='tot', help='dataset name')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--backend', type=str, default='gpt-4-1106-preview', help='gpt model')
    parser.add_argument('--temperature', type=float, default=0.7, help='gpt model temperature')
    parser.add_argument('--sim', type=str, default='naive_score', help='similarity metrics')
    parser.add_argument('--plan', type=str, default='silhouette_score', help='cluster metrics')
    parser.add_argument('--save', type=str, default='/path/to/save', help='thought file path')
    parser.add_argument('--load', type=str, default='', help='thought file path')
    parser.add_argument('--words', type=str, default='/path/to/words', help='words file path')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    main(args)

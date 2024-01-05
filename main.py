import argparse
import os
import random
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
from src.vpt.models.vit_models import ViT
from src.vpt.models.mlp import MLP
from src.vpt.configs.config import get_cfg
from src.solver.loss import PsychoCrossEntropy
from src.solver.lr_scheduler import make_scheduler
from src.solver.optimizer import make_optimizer
from utils.conf import get_world_size
from utils.util import get_coarse_num, AverageMeter, accuracy


def eval(model, val_loader, node_dict, label_to_wnid, label_to_id, device, tot):
    init_weight(node_dict['fall11'], 0)
    solve(model, val_loader, node_dict, label_to_wnid, label_to_id, device, tot, 8)


def train_one_batch(x, model, thought, num_classes, device):
    outputs = torch.zeros([x.shape[0], num_classes], dtype=torch.float32).to(device)
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
                plans.append((plan_w, ts))

        for plan_w, ts in plans:
            # choose a plan and calculate score
            out = torch.zeros([x.shape[0], len(plan_w)], dtype=torch.float32).to(device)
            for i in range(len(plan_w)):
                y = plan_w[i].unsqueeze(0)
                out[:, i] = torch.matmul(x, y.T).squeeze()
            out = out.softmax(dim=1)
            for j in range(len(ts)):
                ts[j].score = out[:, j] * t.score
                thoughts.append(ts[j])
    return outputs


def train(tot, model, criterion, optimizer, scheduler, train_loader, num_classes, total_epoch, device, mode="tot"):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    losses = AverageMeter()
    batch_time = AverageMeter()
    acc = torch.zeros(2).to(device)

    if mode == "baseline":
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
                outputs = train_one_batch(x, model, tot.root, num_classes, device)
            elif mode == "baseline":
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    model = ViT(cfg)

    optimizer = make_optimizer([model], cfg.SOLVER)
    scheduler = make_scheduler(optimizer, cfg.SOLVER)

    criterion = PsychoCrossEntropy(args.classes)

    # 图片统一 224*224（考虑一下32*32）
    train_loader = create_train_dataloader(args)
    val_loader = create_val_dataloader(args)

    a = model.state_dict()
    node_dict, label_to_wnid, label_to_id, labels, _ = build_tree(args, val_loader.dataset.class_to_idx, model.state_dict()['head.last_layer.weight'])

    sim_func = getattr(metrics, args.sim)
    plan_func = getattr(metrics, args.plan)
    tot = ToT(plan_func, sim_func)
    tot.load(args.load, labels)

    num_coarses = get_coarse_num(tot.root, args.classes)
    # 这里可以考虑一下是否固定叶子的weight，直接用预训练的参数还是重新训练
    # 这是不固定，重新训练
    model.head = MLP(
        input_dim=model.feat_dim,
        mlp_dims=[model.feat_dim] * cfg.MODEL.MLP_NUM + [num_coarses+args.classes],
        special_bias=True
    )
    model = model.to(device)

    if get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # find_unused_parameters=True
        )

    train(tot, model, criterion, optimizer, scheduler, train_loader, args.classes, args.epochs, device, mode="tot")

    path_manager = PathManager()
    path_manager.register_handler(HTTPURLHandler())
    save_file = os.path.join(cfg.OUTPUT_DIR, "base_cifar10.pth")
    data = {"model": model.state_dict()}
    with path_manager.open(save_file, "wb") as f:
        torch.save(data, cast(IO[bytes], f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification with gpt')

    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--pretrained', action= "store_true", help = "")
    parser.add_argument('--hier', type=str, default='./structure_released.xml', help='wordnet structure')
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
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    main(args)

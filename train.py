import os
import time
import tqdm
import shutil
from iopath.common.file_io import HTTPURLHandler, PathManager
from typing import cast, IO

import torch
import torch.nn as nn
import torch.utils.data

from utils.conf import is_main_process
from utils.util import AverageMeter, accuracy, reduce_mean
from eval import eval


def compute_scores(x, tot):
    score_dict = {}
    # TODO 检查一下tot.thought_cache全不全
    for k, caches in tot.thought_cache.items():
        score_dict[k] = {}
        for j, cache in caches.items():
            scores = x[:, cache["tids"]]
            score_dict[k][j] = scores.softmax(dim=1)
            # score_dict[k][j] = (indices, torch.LongTensor(coarse_targets).to(targets.device), scores, cache)
    return score_dict


def compute_penalty(outputs, targets, criterion, num_classes, tot):
    penalty = 0.
    for k, caches in tot.thought_cache.items():
        for j, cache in caches.items():
            coarse_targets = []
            indices = []
            for i in range(targets.shape[0]):
                if targets[i].item() in cache["coarse_targets"]:
                    coarse_targets.append(cache["coarse_targets"][targets[i].item()])
                    indices.append(i)

            # 仅对每一层计算loss，不考虑路径概率
            # TODO 在外层计算loss，考虑路径概率
            if len(indices):
                coarse_outputs = outputs[indices, :]
                coarse_outputs = coarse_outputs[:, cache["tids"]]
                coarse_targets = torch.LongTensor(coarse_targets).to(targets.device)
                if cache["do_loss"]:
                    penalty += criterion(coarse_outputs, coarse_targets, num_classes=len(cache["tids"]), norm=True)
                    if torch.isnan(penalty).any():
                        print("Nan error occurs, please check the values computing loss")
                        exit(0)
    penalty += criterion(outputs[:, :num_classes], targets, norm=True)
    return outputs[:, :num_classes], penalty


def train_one_batch(x, targets, criterion, tot, num_classes, device):
    score_dict = compute_scores(x, tot)

    outputs = torch.zeros([x.shape[0], tot.num_coarses], dtype=torch.float32).to(device)
    tot.root.score = torch.FloatTensor([1]).to(device)
    tot.root.path_score = torch.FloatTensor([1]).to(device)

    thoughts = [tot.root]

    while len(thoughts):
        t = thoughts.pop()

        label_list = list(t.labels.keys())
        label_list.sort()
        label_str = str(label_list)[1:-1]
        for i, ts in t.plans.items():
            if ts[0].is_valid():
                for j in range(len(ts)):
                    score = score_dict[label_str][i][:, j]
                    ts[j].path_score = score * t.path_score
                    outputs[:, ts[j].tid] += ts[j].path_score
                    thoughts.append(ts[j])

    outputs, penalty = compute_penalty(outputs, targets, criterion, num_classes, tot)
    return outputs, penalty


def train(cfg, tot, model, criterion, optimizer, scheduler, train_loader, val_loader, num_classes, total_epoch, alpha, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    losses = AverageMeter()
    batch_time = AverageMeter()

    data_len = len(train_loader.dataset)
    save_file = os.path.join(cfg.OUTPUT_DIR, f"{cfg.METHOD}_{cfg.DATA.NAME}-{cfg.K}.pth")
    save_best = os.path.join(cfg.OUTPUT_DIR, f"{cfg.METHOD}_{cfg.DATA.NAME}-{cfg.K}_best.pth")
    train_acc = torch.zeros(2).to(device)
    best_acc = 0

    if cfg.METHOD == "vit":
        criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.start_epoch, total_epoch):
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
                x, corase_x = model(x, return_feature=True)
                x = torch.cat([x, corase_x], dim=1)
                outputs, loss = train_one_batch(x, targets, criterion, tot, num_classes, device)
                # loss += criterion(outputs, targets, norm=True)
            else:
                outputs = model(x)
                loss = criterion(outputs, targets)

            train_acc1, train_acc2 = accuracy(outputs, targets, topk=(1, 5))
            train_acc[0] += train_acc1
            train_acc[1] += train_acc2

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

        train_acc = reduce_mean(train_acc, average=False)
        train_acc = train_acc / data_len

        if is_main_process():
            print(f'\
                train top1: {train_acc[0].item()}\t\
                train top5: {train_acc[1].item()}')

        # eval
        eval_acc = eval(cfg, tot, model, val_loader, alpha, device)
        top1 = eval_acc[0].item()
        is_best = top1 > best_acc
        best_acc = max(top1, best_acc)

        if is_main_process():
            state = {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }
            if cfg.NUM_GPUS > 1:
                state = {"model": model.module.state_dict()}
            else:
                state = {"model": model.state_dict()}

            torch.save(state, save_file)
            if is_best:
                shutil.copyfile(save_file, save_best)

        if cfg.NUM_GPUS > 1:
            torch.distributed.barrier()

import os
import time
import tqdm
from iopath.common.file_io import HTTPURLHandler, PathManager
from typing import cast, IO

import torch
import torch.nn as nn
import torch.utils.data

from utils.conf import is_main_process
from utils.util import AverageMeter, accuracy, reduce_mean


def compute_penalty(x, corase_feat, targets, criterion, tot):
    penalty = 0.
    score_dict = {}

    for k, _ in tot.plan_dict.items():
        plans = []
        do_loss = []
        for ts in _.values():
            if ts[0].is_valid():
                plan_w = []
                for i in range(len(ts)):
                    if ts[i].stop():
                        w = x[:, ts[i].tid]
                    else:
                        w = corase_feat[:, ts[i].tid]
                    plan_w.append(w)

                coarse_targets = torch.ones_like(targets) * (len(ts)-1)
                if ts[len(ts)-1].name == "Other":
                    do_loss.append(True)
                    for i in range(targets.shape[0]):
                        for j in range(len(ts)):
                            if targets[i].item() in ts[j].labels:
                                coarse_targets[i] = j
                                break
                else:
                    do_loss.append(False)

                plans.append((plan_w, ts, coarse_targets))

        score_dict[k] = []
        for i in range(len(plans)):
            plan_w, ts, coarse_targets = plans[i]
            coarse_x = torch.stack(plan_w).T
            coarse_out = coarse_x.softmax(dim=1)
            if do_loss[i]:
                penalty += criterion(coarse_out, coarse_targets, num_classes=len(ts))
            score_dict[k].append(coarse_out)
    return penalty, score_dict


def train_one_batch(x, corase_x, targets, criterion, tot, num_classes, device):
    penalty, score_dict = compute_penalty(x, corase_x, targets, criterion, tot)

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
                x, corase_x = model(x, return_feature=True)
                outputs, loss = train_one_batch(x, corase_x, targets, criterion, tot, num_classes, device)
                # loss += criterion(outputs, targets, norm=True)
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

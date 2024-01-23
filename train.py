import os
import time
import tqdm
import shutil

import torch
import torch.nn as nn
import torch.utils.data

from utils.conf import is_main_process
from utils.util import AverageMeter, accuracy, reduce_mean
from eval import eval


def compute_wrong_acc(outputs, targets, num_classes):
    wrong_acc = torch.zeros(num_classes).to(outputs.device)
    pred = outputs.data.max(1)[1]
    wrong_indices = torch.nonzero(pred.data!=targets.data)
    wrong_targets = targets[wrong_indices]
    for i in range(num_classes):
        wrong_acc[i] += (wrong_targets.data==i).sum()
    return wrong_acc


def compute_penalty(x, targets, criterion, tot):
    penalty = 0.
    score_dict = {}
    coarse_acc = {}
    # TODO 检查一下tot.thought_cache全不全
    for k, caches in tot.thought_cache.items():
        score_dict[k] = {}
        coarse_acc[k] = {}
        for j, cache in caches.items():
            coarse_targets = []
            indices = []
            for i in range(targets.shape[0]):
                if targets[i].item() in cache["coarse_targets"]:
                    coarse_targets.append(cache["coarse_targets"][targets[i].item()])
                    indices.append(i)
            
            # coarse_acc[k][j] = {}
            # for i in cache["tids"]:
            #     coarse_acc[k][j][i] = [torch.LongTensor([0]).to(targets.device).sum(), torch.LongTensor([0]).to(targets.device).sum()]

            if len(indices):
                coarse_x = x[indices, :]
                coarse_x = coarse_x[:, cache["tids"]]
                coarse_out = coarse_x.softmax(dim=1)
                coarse_targets = torch.LongTensor(coarse_targets).to(targets.device)
                if cache["do_loss"]:
                    penalty += criterion(coarse_x, coarse_targets, cache["samples_per_cls"], num_classes=len(cache["tids"]))
                    if torch.isnan(penalty).any():
                        print("Nan error occurs, please check the values computing loss")
                        exit(0)
                # coarse_pred = coarse_out.data.max(1)[1]
                coarse_acc[k][j] = compute_wrong_acc(coarse_out, coarse_targets, len(cache["tids"]))
                # coarse_acc[k][j] = [coarse_pred.eq(coarse_targets.data).sum(), torch.LongTensor([len(indices)]).to(targets.device).sum()]
            else:
                coarse_acc[k][j] = torch.zeros(len(cache["tids"])).to(x.device)
                # coarse_acc[k][j] = [torch.LongTensor([0]).to(targets.device).sum(), torch.LongTensor([0]).to(targets.device).sum()]
            out = x[:, cache["tids"]].softmax(dim=1)
            score_dict[k][j] = out
    return penalty, score_dict, coarse_acc


def train_one_batch(x, targets, criterion, tot, num_classes, device):
    penalty, score_dict, coarse_acc = compute_penalty(x, targets, criterion, tot)

    scores = torch.zeros([x.shape[0], num_classes], dtype=torch.float32).to(device)
    outputs = torch.zeros([x.shape[0], num_classes], dtype=torch.float32).to(device)
    tot.root.score = torch.FloatTensor([1]).to(device)
    tot.root.path_score = torch.FloatTensor([1]).to(device)

    thoughts = [tot.root]
    while len(thoughts):
        t = thoughts.pop()
        if t.stop():
            label_id = t.tid
            scores[:, label_id] += x[:, label_id]
            outputs[:, label_id] += t.path_score

        label_list = list(t.labels.keys())
        label_list.sort()
        label_str = str(label_list)[1:-1]
        for i, ts in t.plans.items():
            if ts[0].is_valid():
                for j in range(len(ts)):
                    score = score_dict[label_str][i][:, j]
                    ts[j].path_score = score * t.path_score
                    thoughts.append(ts[j])

    leaves_cnt = torch.FloatTensor(tot.leaves_cnt).to(device)
    scores = scores / leaves_cnt
    outputs = outputs / leaves_cnt

    return scores, penalty, coarse_acc


def train(cfg, tot, model, criterion, optimizer, scheduler, train_loader, val_loader, num_classes, total_epoch, alpha, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    losses = AverageMeter()
    batch_time = AverageMeter()

    data_len = len(train_loader.dataset)
    classes = train_loader.dataset.classes
    img_num_list = train_loader.dataset.img_num_list

    save_file = os.path.join(cfg.OUTPUT_DIR, f"{cfg.METHOD}_{cfg.DATA.NAME}-{cfg.K}_{cfg.loss}.pth")
    save_best = os.path.join(cfg.OUTPUT_DIR, f"{cfg.METHOD}_{cfg.DATA.NAME}-{cfg.K}_{cfg.loss}_best.pth")

    for epoch in range(cfg.start_epoch, total_epoch):
        wrong_acc = torch.zeros(num_classes).to(device)
        coarse_acc = None
        train_acc = torch.zeros(2).to(device)
        best_acc = 0
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
                outputs, loss, acc = train_one_batch(x, targets, criterion, tot, num_classes, device)
                loss += criterion(outputs, targets)
                outputs = outputs.softmax(dim=1)
                # loss += criterion(outputs, targets, norm=True)

                if coarse_acc == None:
                    coarse_acc = acc
                else:
                    for k in acc.keys():
                        for j in acc[k].keys():
                            coarse_acc[k][j] += acc[k][j]
            else:
                outputs = model(x)
                loss = criterion(outputs, targets)
                outputs = outputs.softmax(dim=1)

            train_acc1, train_acc2 = accuracy(outputs, targets, topk=(1, 5))
            train_acc[0] += train_acc1
            train_acc[1] += train_acc2

            wrong_acc_t = compute_wrong_acc(outputs, targets, num_classes)
            wrong_acc += wrong_acc_t

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
        if cfg.METHOD == "tot":
            for k in acc.keys():
                for j in acc[k].keys():
                    coarse_acc[k][j] = reduce_mean(coarse_acc[k][j], average=False)
                    coarse_acc[k][j] /= torch.FloatTensor(tot.thought_cache[k][j]["samples_per_cls"]).to(device)
                    # coarse_acc[k][j][1] = reduce_mean(coarse_acc[k][j][1], average=False).item()
            coarse_acc_t = {}
            for k in coarse_acc.keys():
                name = tot.plan_dict[k][0][0].parent.name
                coarse_acc_t[name] = {}
                for j in coarse_acc[k].keys():
                    coarse_acc_t[name][str(tot.name_cache[k][j])] = list(coarse_acc[k][j].cpu().detach().numpy())

        wrong_acc = reduce_mean(wrong_acc, average=False)

        if is_main_process():
            if cfg.METHOD == "tot":
                print(f'\
                    train top1: {train_acc[0].item()}\t\
                    train top5: {train_acc[1].item()}\n\
                    coarse acc:\n{coarse_acc_t}')
            else:
                print(f'\
                    train top1: {train_acc[0].item()}\t\
                    train top5: {train_acc[1].item()}')
            for i in range(num_classes):
                print(f"{classes[i]}: ({wrong_acc[i].item()}, {wrong_acc[i].item() / img_num_list[i]})")

        # eval
        eval_acc = eval(cfg, tot, model, val_loader, num_classes, alpha, device)
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
                state["model"] = model.module.state_dict()
            else:
                state["model"] = model.state_dict()

            torch.save(state, save_file)
            if is_best:
                shutil.copyfile(save_file, save_best)

        if cfg.NUM_GPUS > 1:
            torch.distributed.barrier()
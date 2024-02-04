import progressbar
import torch
import torch.utils.data

from utils.conf import is_main_process
from utils.util import accuracy, reduce_mean


def compute_wrong_acc(outputs, targets, num_classes):
    wrong_acc = torch.zeros(num_classes).to(outputs.device)
    pred = outputs.data.max(1)[1]
    wrong_indices = torch.nonzero(pred.data!=targets.data)
    wrong_targets = targets[wrong_indices]
    for i in range(num_classes):
        wrong_acc[i] += (wrong_targets.data==i).sum()
    return wrong_acc


def eval_one_batch(x, targets, tot, num_classes, device):
    score_dict = {}
    coarse_acc = {}
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
            if len(indices):
                coarse_x = x[indices, :]
                coarse_x = coarse_x[:, cache["tids"]]
                coarse_out = coarse_x.softmax(dim=1)
                coarse_targets = torch.LongTensor(coarse_targets).to(targets.device)
                coarse_acc[k][j] = compute_wrong_acc(coarse_out, coarse_targets, len(cache["tids"]))
            else:
                coarse_acc[k][j] = torch.zeros(len(cache["tids"])).to(x.device)
            out = x[:, cache["tids"]].softmax(dim=1)
            score_dict[k][j] = out

    scores = torch.zeros([x.shape[0], num_classes], dtype=torch.float32).to(device)
    outputs = torch.zeros([x.shape[0], num_classes], dtype=torch.float32).to(device)
    tot.root.score = torch.FloatTensor([1]).to(device)
    tot.root.path_score = torch.FloatTensor([1]).to(device)

    thoughts = [tot.root]
    while len(thoughts):
        t = thoughts.pop()
        if t.stop():
            label_id = t.tid
            scores[:, label_id] += t.score
            # scores[:, label_id] += x[:, label_id]
            # outputs[:, label_id] += t.path_score

        label_list = list(t.labels.keys())
        label_list.sort()
        label_str = str(label_list)[1:-1]
        for i, ts in t.plans.items():
            if ts[0].is_valid():
                for j in range(len(ts)):
                    score = score_dict[label_str][i][:, j]
                    ts[j].path_score = score * t.path_score
                    thoughts.append(ts[j])

                    if t.tid == -1:
                        ts[j].score = x[:, ts[j].tid]
                    else:
                        ts[j].score = (x[:, ts[j].tid] + t.score) / 2

    leaves_cnt = torch.FloatTensor(tot.leaves_cnt).to(device)
    scores = scores / leaves_cnt
    # outputs = outputs / leaves_cnt

    return scores, coarse_acc


@torch.no_grad()
def eval(cfg, tot, model, val_loader, num_classes, alpha, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_len = len(val_loader.dataset)
    classes = val_loader.dataset.classes
    img_num_list = val_loader.dataset.img_num_list

    eval_acc = torch.zeros(2).to(device)
    wrong_acc = torch.zeros(num_classes).to(device)
    coarse_acc = None
    wrong_info = {}
    cnt = [0 for _ in range(num_classes)]

    for i in range(num_classes):
        wrong_info[i] = torch.zeros([num_classes]).to(device)

    if is_main_process():
        bar = progressbar.ProgressBar(0, len(val_loader))

    for idx, (x, targets) in enumerate(val_loader):
        x = x.to(device)
        targets = targets.to(device)

        if cfg.METHOD == "tot":
            tot.clean()
            x, corase_x = model(x, return_feature=True)
            x = torch.cat([x, corase_x], dim=1)
            # outputs, acc = eval_one_batch(x, targets, tot, num_classes, device)
            # outputs = outputs.softmax(dim=1)
            # pred = outputs.max(1)[1]
            # if coarse_acc == None:
            #     coarse_acc = acc
            # else:
            #     for k in acc.keys():
            #         for j in acc[k].keys():
            #             coarse_acc[k][j] += acc[k][j]

            pred, c = tot.solve(x, targets, alpha, method='dfs')
            for i in range(num_classes):
                cnt[i] += c[i]
            eval_acc[0] += pred.eq(targets.data).sum()

        elif cfg.METHOD == "vit":
            outputs = model(x)
            outputs = outputs.softmax(dim=1)
            acc1, acc2 = accuracy(outputs, targets, topk=(1, 5))
            eval_acc[0] += acc1
            eval_acc[1] += acc2
            pred = outputs.max(1)[1]

        wrong_indices = torch.nonzero(pred.data!=targets.data)
        wrong_targets = targets[wrong_indices]
        for i in range(num_classes):
            wrong_acc[i] += (wrong_targets.data==i).sum()

        for i in range(pred.shape[0]):
            if pred[i] != targets[i]:
                wrong_info[targets[i].item()][pred[i].item()] += 1

        if cfg.NUM_GPUS > 1:
            torch.distributed.barrier()

        if is_main_process():
            bar.update(idx+1)

    # if cfg.METHOD == "tot":
    #     for k in acc.keys():
    #         for j in acc[k].keys():
    #             coarse_acc[k][j] = reduce_mean(coarse_acc[k][j], average=False)
    #             coarse_acc[k][j] /= torch.FloatTensor(tot.thought_cache[k][j]["samples_per_cls"]).to(device)
    #     coarse_acc_t = {}
    #     for k in coarse_acc.keys():
    #         name = tot.plan_dict[k][0][0].parent.name
    #         coarse_acc_t[name] = {}
    #         for j in coarse_acc[k].keys():
    #             coarse_acc_t[name][str(tot.name_cache[k][j])] = list(coarse_acc[k][j].cpu().detach().numpy())

    wrong_acc = reduce_mean(wrong_acc, average=False)
    eval_acc = reduce_mean(eval_acc, average=False)
    eval_acc = eval_acc / data_len

    for i in range(num_classes):
        wrong_info[i] = reduce_mean(wrong_info[i], average=False)

    if is_main_process():
        print(f'\
            val top1: {eval_acc[0].item()}\t\
            val top5: {eval_acc[1].item()}')
        print("hard to classify number: ", cnt)
        # if cfg.METHOD == "tot":
        #     print(f'coarse acc:\n{coarse_acc_t}')
        for i in range(num_classes):
            print(f"{classes[i]}: ({wrong_acc[i].item()}, {wrong_acc[i].item() / img_num_list[i]})")
        print("--------------")
        for i in range(num_classes):
            w_v, w_i = wrong_info[i].topk(2)
            print(f"{classes[i]}: {classes[w_i[0].item()]}: {w_v[0].item()} times, {classes[w_i[1].item()]}: {w_v[1].item()} times")
    return eval_acc

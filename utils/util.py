import torch.distributed as dist
import torch
from utils.conf import get_world_size


class Result:
    def __init__(self, name, status, score, parent=None) -> None:
        self.name = name
        self.status = status
        self.parent = parent
        self.score = score
        self.children = []

    def add(self, r):
        self.children.append(r)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def reduce_mean(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)   # 对不同设备之间的value求和
        if average:  # 如果需要求平均，获得多块GPU计算loss的均值
            value /= world_size
    return value


def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res

def get_coarse_labels(root):
    ts = [root]
    coarse = dict()
    cnt = 1
    while len(ts):
        t = ts.pop()
        if len(t.labels) != 1:
            if t.name[-1] not in coarse:
                coarse[t.name[-1]] = []
            coarse[t.name[-1]].append(t.labels)
            for _ in t.plans.values():
                for child in _:
                    ts.append(child)
                    cnt += 1
    return coarse


def get_coarse_num(root):
    num_coarse = -1
    num_fine = 0

    ts = [root]
    while len(ts):
        t = ts.pop()
        if t.stop():
            continue
        t.tid = num_coarse
        num_coarse += 1
        for _ in t.plans.values():
            for child in _:
                ts.insert(0, child)

    return num_coarse

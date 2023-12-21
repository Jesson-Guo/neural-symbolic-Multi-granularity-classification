import torch
import torch.nn.functional as F
from torchmetrics.clustering import DunnIndex


def kl_divergence(x, cluster, reduction='batchmean'):
    kl = []
    for _, weights in cluster.items():
        y = torch.stack(weights).mean(dim=0)
        out = F.kl_div(x, y, reduction=reduction)
        kl.append(out)
    return kl


def cosine_similarity(x, cluster):
    cs = []
    for _, weights in cluster.items():
        y = torch.stack(weights).mean(dim=0)
        out = F.cosine_similarity(x, y)
        cs.append(out)
    return cs


def dunn_index(clusters, p=2):
    data = []
    labels = []

    i = 0
    for _, ws in clusters.items():
        for v in ws:
            data.append(v)
            labels.append(i)

    data = torch.stack(data)
    labels = torch.tensor(labels)
    return DunnIndex(p=p)(data, labels)

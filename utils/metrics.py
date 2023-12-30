import torch
import torch.nn.functional as F
from torchmetrics.clustering import DunnIndex, DaviesBouldinScore, CalinskiHarabaszScore
from pytorch_adapt.layers import SilhouetteScore


def clusters_to_xy(clusters):
    x, y = [], []
    i = 0
    for _, weights in clusters.items():
        for w in weights:
            x.append(w.data)
            y.append(i)
        i += 1
    x = torch.stack(x)
    y = torch.LongTensor(y).to(x.device)
    return x, y


def naive_score(x, cluster):
    ns = []
    for _, weights in cluster.items():
        y = torch.stack(weights).mean(dim=0)
        out = torch.dot(x, y.T)
        ns.append(-out)
        del y
    return torch.stack(ns)


def kl_divergence(x, cluster, reduction='batchmean'):
    kl = []
    for _, weights in cluster.items():
        y = torch.stack(weights).mean(dim=0)
        out = F.kl_div(x, y, reduction=reduction)
        kl.append(out)
    return torch.stack(kl)


def cosine_similarity(x, cluster):
    cs = []
    for _, weights in cluster.items():
        y = torch.stack(weights).mean(dim=0)
        out = F.cosine_similarity(x, y)
        cs.append(out)
    return torch.stack(cs)


def dunn_index(clusters, p=2):
    x, y = clusters_to_xy(clusters)
    di = DunnIndex(p=p)
    out = di(x, y)
    # 越大越好
    return out


def silhouette_score(clusters):
    x, y = clusters_to_xy(clusters)
    unique_labels = torch.unique(y)
    num_samples = len(x)
    if 1 == len(unique_labels):
        return 0
    if len(unique_labels) == num_samples:
        return 2

    if x.shape[0] == 2:
        return 2
    ss = SilhouetteScore()
    out = ss(x, y)
    del x, y

    if out < -0.5:
        return 0
    elif out > 0.5:
        return 2
    else:
        return 1


def db_index(clusters):
    x, y = clusters_to_xy(clusters)
    dbi = DaviesBouldinScore()
    # 越大越好
    return dbi(x, y)


def ch_score(clusters):
    x, y = clusters_to_xy(clusters)
    dbi = CalinskiHarabaszScore()
    # 越大越好
    return dbi(x, y)

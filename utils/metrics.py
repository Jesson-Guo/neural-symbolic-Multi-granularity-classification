import torch
import torch.nn.functional as F
from torchmetrics.clustering import DunnIndex, DaviesBouldinScore, CalinskiHarabaszScore
from pytorch_adapt.layers import SilhouetteScore


def clusters_to_xy(cluster):
    x, y = [], []
    i = 0
    for _, weights in cluster.items():
        for w in weights:
            x.append(w)
            y.append(i)
        i += 1
    return x, y


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


def dunn_index(cluster, p=2):
    x, y = clusters_to_xy(cluster)
    di = DunnIndex(p=p)
    out = di(x, y)
    # 越大越好
    return out


def silhouette_score(cluster):
    x, y = clusters_to_xy(cluster)
    ss = SilhouetteScore()
    out = ss(x, y)

    if out < -0.5:
        return 0
    elif out > 0.5:
        return 2
    else:
        return 1


def db_index(cluster):
    x, y = clusters_to_xy(cluster)
    dbi = DaviesBouldinScore()
    # 越大越好
    return dbi(x, y)


def ch_score(cluster):
    x, y = clusters_to_xy(cluster)
    dbi = CalinskiHarabaszScore()
    # 越大越好
    return dbi(x, y)

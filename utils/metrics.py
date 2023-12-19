import torch.nn.functional as F


def kl_divergence(x, y, reduction='batchmean'):
    return F.kl_div(x, y, reduction=reduction)


def cosine_similarity(x, y):
    return F.cosine_similarity(x, y)

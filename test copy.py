import torch.nn.functional as F
import torch.nn as nn
import torch
import openai
import os
import numpy as np
from torchmetrics.clustering import DunnIndex

from src.gpt import GPT
import json


a = {'Plan1': {'StorageContainers': [412, 427, 463, 478, 588, 692, 719, 737, 748, 756, 771, 790, 797, 868, 876, 883, 893, 896, 898, 899, 900, 907], 'DrinkContainers': [737, 756, 898, 899, 900, 907]}, 'Plan2': {'SmallItems': [412, 719, 748, 893, 897], 'LargeContainers': [427, 463, 478, 588, 692, 756, 771, 790, 797, 868, 876, 883, 896, 898, 899, 900, 907]}}

b = [412, 427, 441, 463, 478, 492, 519, 572, 588, 653, 666, 692, 709, 719, 720, 725, 728, 737, 738, 748, 756, 771, 790, 804, 868, 876, 883, 893, 896, 898, 899, 900, 901, 907]

plans = []
for i, plan in a.items():
    s = set()
    for ls in plan.values():
        for l in ls:
            s.add(l)
    plans.append(s)

remain = []
for s in plans:
    r = []
    for i in b:
        if not i in s:
            r.append(i)
    remain.append(r)

print(remain)

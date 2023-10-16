import os
import numpy as np
import cv2
from typing import List, Tuple, Dict

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from src.hierarchy import label2id


class TinyImagenet200(Dataset):
    """Tiny imagenet dataloader"""

    dataset = None

    def __init__(self, root="./data", image_size=64, transform=None, train=True):
        super().__init__()
        dataset = _TinyImagenet200Train if train else _TinyImagenet200Val
        self.root = root
        self.dataset = dataset(root, transform)
        self.classes = self.dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class _TinyImagenet200Train(torchvision.datasets.ImageFolder):
    def __init__(self, root="./data", transform=None):
        super().__init__(os.path.join(root, "tiny-imagenet-200/train"), transform=transform)


class _TinyImagenet200Val(torchvision.datasets.ImageFolder):
    def __init__(self, root="./data", transform=None):
        super().__init__(os.path.join(root, "tiny-imagenet-200/val"), transform=transform)


class Imagenet1000(Dataset):
    """ImageNet dataloader"""

    dataset = None

    def __init__(self, root="./data", image_size=224, transform=None, train=True):
        super().__init__()
        dataset = _Imagenet1000Train if train else _Imagenet1000Val
        self.root = root
        self.dataset = dataset(root, transform)
        self.classes = self.dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class _Imagenet1000Train(torchvision.datasets.ImageFolder):
    def __init__(self, root="./data", *args, **kwargs):
        super().__init__(os.path.join(root, "imagenet-1k/train"), *args, **kwargs)


class _Imagenet1000Val(torchvision.datasets.ImageFolder):
    def __init__(self, root="./data", *args, **kwargs):
        super().__init__(os.path.join(root, "imagenet-1k/val"), *args, **kwargs)


def create_dataloader(args, dataset, sampler=None, training=False):
    if training:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            num_workers=args.workers,
            pin_memory=True
        )
        return train_loader

    else:
        val_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        return val_loader

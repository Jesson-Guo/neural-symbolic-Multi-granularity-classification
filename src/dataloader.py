import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.conf import get_world_size


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
        super().__init__(os.path.join(root, "imagenet/images/train"), *args, **kwargs)


class _Imagenet1000Val(torchvision.datasets.ImageFolder):
    def __init__(self, root="./data", *args, **kwargs):
        super().__init__(os.path.join(root, "imagenet/images/val"), *args, **kwargs)


def create_train_dataloader(cfg):
    if cfg.DATA.NAME == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=cfg.DATA.ROOT,
            train=True,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        )
    elif cfg.DATA.NAME == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=cfg.DATA.ROOT,
            train=True,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        )
    elif cfg.DATA.NAME == "tiny-imagenet":
        train_dataset = TinyImagenet200(
            root=cfg.DATA.ROOT,
            image_size=64,
            transform=transforms.Compose([
                transforms.RandomCrop(64, 8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
            ]),
            train=True
        )
    elif cfg.DATA.NAME == "imagenet":
        train_dataset = Imagenet1000(
            root=cfg.DATA.ROOT,
            image_size=224,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]),
            train=True
        )

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.DATA.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    return train_loader


def create_val_dataloader(cfg):
    if cfg.DATA.NAME == "cifar10":
        val_dataset = torchvision.datasets.CIFAR10(
            root=cfg.DATA.ROOT,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )
    elif cfg.DATA.NAME == "cifar100":
        val_dataset = torchvision.datasets.CIFAR100(
            root=cfg.DATA.ROOT,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )
    elif cfg.DATA.NAME == "tiny-imagenet":
        val_dataset = TinyImagenet200(
            root=cfg.DATA.ROOT,
            image_size=64,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
            ]),
            train=False
        )
    elif cfg.DATA.NAME == "imagenet":
        val_dataset = Imagenet1000(
            root=cfg.DATA.ROOT,
            image_size=224,
            transform=transforms.Compose([
                transforms.Resize(224 + 32),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]),
            train=False
        )

    val_sampler = None
    if get_world_size() > 1:
        val_sampler = DistributedSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=cfg.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    return val_loader

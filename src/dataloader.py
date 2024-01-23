import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.conf import get_world_size


class CIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root, train, transform, target_transform, download)
        num = len(self.data) / self.cls_num
        self.img_num_list = [num for _ in range(self.cls_num)]


class CIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100, self).__init__(root, train, transform, target_transform, download)
        num = len(self.data) / self.cls_num
        self.img_num_list = [num for _ in range(self.cls_num)]


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(self.img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


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
        train_dataset = CIFAR10(
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
    elif cfg.DATA.NAME == "cifar10-lt":
        train_dataset = IMBALANCECIFAR10(
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
        train_dataset = CIFAR100(
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
    elif cfg.DATA.NAME == "cifar100-lt":
        train_dataset = IMBALANCECIFAR100(
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
        val_dataset = CIFAR10(
            root=cfg.DATA.ROOT,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        )
    elif cfg.DATA.NAME == "cifar10-lt":
        val_dataset = IMBALANCECIFAR10(
            root=cfg.DATA.ROOT,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        )
    elif cfg.DATA.NAME == "cifar100":
        val_dataset = CIFAR100(
            root=cfg.DATA.ROOT,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )
    elif cfg.DATA.NAME == "cifar100-lt":
        val_dataset = IMBALANCECIFAR100(
            root=cfg.DATA.ROOT,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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

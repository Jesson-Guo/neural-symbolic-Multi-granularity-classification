import os
import numpy as np
import cv2
from typing import List, Tuple, Dict

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from src.hierarchy import label2id


# class TinyImagenet200(Dataset):
#     """Tiny imagenet dataloader"""

#     dataset = None

#     def __init__(self, root="./data", image_size=64, transform=None, train=True):
#         super().__init__()
#         dataset = _TinyImagenet200Train if train else _TinyImagenet200Val
#         self.root = root
#         self.image_size = image_size
#         self.transform = transform
#         self.dataset = dataset(root, *args, **kwargs)
#         self.classes = self.dataset.classes
#         self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

#     def __getitem__(self, i):
#         return self.dataset[i]

#     def __len__(self):
#         return len(self.dataset)


# class _TinyImagenet200Train(torchvision.datasets.ImageFolder):
#     def __init__(self, root="./data", *args, **kwargs):
#         super().__init__(os.path.join(root, "tiny-imagenet-200/train"), *args, **kwargs)


# class _TinyImagenet200Val(torchvision.datasets.ImageFolder):
#     def __init__(self, root="./data", *args, **kwargs):
#         super().__init__(os.path.join(root, "tiny-imagenet-200/val"), *args, **kwargs)

#         self.path_to_class = {}
#         with open(os.path.join(self.root, "val_annotations.txt")) as f:
#             for line in f.readlines():
#                 parts = line.split()
#                 path = os.path.join(self.root, "images", parts[0])
#                 self.path_to_class[path] = parts[1]

#         self.classes = list(sorted(set(self.path_to_class.values())))
#         self.class_to_idx = {label: self.classes.index(label) for label in self.classes}

#     def __getitem__(self, i):
#         sample, _ = super().__getitem__(i)
#         path, _ = self.samples[i]
#         label = self.path_to_class[path]
#         target = self.class_to_idx[label]
#         return sample, target

#     def __len__(self):
#         return super().__len__()


class TinyImagenetDataset(Dataset):
    def __init__(
        self,
        root: str = None,
        image_size: int = 224,
        transform: transforms = None
    ) -> None:
        super().__init__()
        self.image_size = image_size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.instances = self._make_dataset(root)

    def _make_dataset(
        self,
        directory: str
    ) -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
        directory = os.path.expanduser(directory)

        instances = []

        for d in os.listdir(directory):
            # skip directory start with '.'
            if d[0] != '.':
                f = os.path.join(os.path.join(directory, d), d+'_boxes.txt')
                f = open(f, 'r')
                for line in f.readlines():
                    split_line = line.split('\n')
                    split_line = split_line[0].split('\t')
                    img_path = os.path.join(os.path.join(directory, d), 'images/'+split_line[0])
                    instances.append((img_path, label2id[d]))

        return instances

    def __getitem__(self, index):
        img_path, target = self.instances[index]

        image = cv2.imread(img_path)
        # Convert BGR to RGB color format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.instances)


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

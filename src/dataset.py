import os
import numpy as np
from typing import List, Tuple, Dict

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class TinyImagenetDataset(Dataset):
    def __init__(
        self,
        file_path: str = None,
        transform: transforms = None,
        local_rank: int = -1
    ) -> None:
        super().__init__()
        self.local_rank = local_rank

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.label2id = self._make_label_dict(file_path)
        self.instances = self._make_dataset(file_path)

    def _make_label_dict(
        self,
        directory: str
    ) -> Dict[str, int]:
        directory = os.path.expanduser(directory)

        label2id = dict()
        l = 1

        for d in os.listdir(directory):
            # skip directory start with '.'
            if d[0] != '.':
                if d not in label2id:
                    label2id[d] = l
                    l += 1

        return label2id

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
                    xmin, ymin, xmax, ymax = int(split_line[1]), int(split_line[2]), int(split_line[3]), int(split_line[4])
                    if 0 < xmin < xmax and 0 < ymin < ymax:
                        box = (xmin, ymin, xmax, ymax)
                        instances.append((img_path, self.label2id[d], box))

        return instances

    def __getitem__(self, index):
        img_path, label, box = self.instances[index]

        x = Image.open(img_path).convert('RGB')
        x = self.transform(x)
        box = np.array(box, dtype=np.float32)

        return x, label, box

    def __len__(self):
        return len(self.instances)

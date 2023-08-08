import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class TinyImagenetDataset(Dataset):
    def __init__(
        self,
        file_path: str = None,
        transform: transforms = None
    ) -> None:
        super().__init__()
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.instances = self._make_dataset(file_path)
    
    def _make_dataset(
        directory: str,
    ) -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
        directory = os.path.expanduser(directory)

        instances = []

        for root, dirs, files in os.walk(directory):
            for d in dirs:
                f = os.path.join(os.path.join(root, d), d+'_boxes.txt')
                annotations = open(f, 'r')
                for line in f.readlines():
                    split_line = line.split('\n')
                    split_line = split_line[0].split('\t')
                    img_path = os.path.join(os.path.join(root, d), split_line[0])
                    box = (split_line[2], split_line[3], split_line[4], split_line[5])
                    instances.append((img_path, d, box))
        
        return instances

    def __getitem__(self, index):
        img_path, label, box = self.instances[index]
        x = Image.open(img_path).convert('RGB')
        return x, label, box

    def __len__(self):
        return len(self.instances)

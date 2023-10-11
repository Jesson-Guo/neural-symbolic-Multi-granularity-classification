import os
import numpy as np
import cv2
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TinyImagenetDataset(Dataset):
    def __init__(
        self,
        file_path: str = None,
        image_size: int = 224,
        label2id: Dict[str, int] = None,
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
        self.label2id = label2id
        self.instances = self._make_dataset(file_path)

    def _make_label_path(self, label_paths, label2id):
        lpaths = {}
        for k in label_paths.keys():
            lpaths[k] = []
            for a in label_paths[k]:
                lpaths[k].append(label2id[a])
        return lpaths

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
                    # if 0 < xmin < xmax and 0 < ymin < ymax:
                    box = (xmin, ymin, xmax, ymax)
                    instances.append((img_path, self.label2id[d], box))

        return instances
    
    def _resize(self, image):
        h0, w0 = image.shape[:2]
        r = self.image_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)))
        return image

    def _check_box(self, box, width, height):
        """
        Check that all x_max and y_max are not more than the image
        width or height.
        """
        xmin, ymin, xmax, ymax = box
        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width
        if xmax - xmin <= 1.0:
            xmin = xmin - 1
        if ymax - ymin <= 1.0:
            ymin = ymin - 1
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, index):
        img_path, label, box = self.instances[index]

        image = cv2.imread(img_path)
        # Convert BGR to RGB color format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = self._resize(image)
        image /= 255.0
        image_height = image.shape[0]
        image_width = image.shape[1]

        ori_box = self._check_box(box, image_width, image_height)
        xmin, ymin, xmax, ymax = ori_box

        xmin_final = (xmin/image_width)*image.shape[1]
        xmax_final = (xmax/image_width)*image.shape[1]
        ymin_final = (ymin/image_height)*image.shape[0]
        ymax_final = (ymax/image_height)*image.shape[0]

        box = [xmin_final, ymin_final, xmax_final, ymax_final]
        box = self._check_box(box, image_width, image_height)

        image = self.transform(image)
        box = torch.as_tensor(box, dtype=torch.float32)
        # label = torch.as_tensor(label)

        target = {}
        target["boxes"] = box.reshape(1,4)
        target["labels"] = label
        # target["labels"] = label.reshape(1)

        return image, target

    def __len__(self):
        return len(self.instances)

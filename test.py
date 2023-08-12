import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import progressbar

import torch
from PIL import Image
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor

from src.resnet import *


def visualize_box(img, bbox):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255,0,0))
    return img


def test(model, optimizer, scheduler, ckpt_path, test_path, batch, device):
    test_path = os.path.join(test_path, 'test/images')

    # load checkpoint
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_path))

    # load test images
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_path = os.path.join(test_path, 'images')
    test_images = []
    test_files = []

    num_test_file = 0
    for f in os.listdir(test_path):
        f = os.path.join(test_path, f)
        x = Image.open(f).convert('RGB')
        test_images.append(transform(x).to(device))
        test_files.append(f)
        num_test_file += 1

    def partition(l, size):
        return [l[i:i+size] for i in range(0, len(l), size)]

    test_images = partition(test_images, batch)
    test_files = partition(test_files, batch)

    # predict
    print("=====> START PREDICTING")
    bar = progressbar.ProgressBar(0, len(test_images))

    model.eval()
    predictions = []

    for i in range(len(test_images)):
        pred = model(test_images[i])
        predictions.append(pred)
        bar.update(i)

    print("=====> START SAVING OUTPUTS")
    bar = progressbar.ProgressBar(0, num_test_file)

    for i in range(len(predictions)):
        # visualize
        for j in range(batch):
            image = cv2.imread(test_files[i][j])
            bbox = predictions[i][j]['boxes'].detach().numpy().reshape((4))
            image = visualize_box(image, bbox)
            # plt.imshow()
            # plt.show()
            file_name = test_files[i][j].split('\n')
            file_name = file_name[0].split('/')[-1]
            plt.savefig(os.path.join("./output/test", file_name))
            bar.update(i*batch + j)

    print("===== END TEST =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tiny imagenet training')
    parser.add_argument('--ckpt_path', type=str, default='.', help='checkpoint path')
    parser.add_argument('--test_path', type=str, default='.', help='test directory path')
    parser.add_argument('--test_batch', type=int, default=100, help='number of images to test per batch')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if args.local_rank != -1:
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl", init_method='env://')

    backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200+1, use_cbam=args.use_cbam)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=5)
    model = FasterRCNN(backbone=backbone, num_classes=200+1, min_size=64, max_size=64)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)

    cudnn.benchmark = True

    test(model, optimizer, scheduler, args.ckpt_path, args.test_path, args.test_batch, device)

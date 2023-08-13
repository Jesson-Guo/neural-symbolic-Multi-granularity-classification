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
    for i in range(bbox.shape[0]):
        x_min, y_min, x_max, y_max = bbox[i].numpy().reshape((4))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0,255,0))
    return img


@torch.no_grad()
def test(model, optimizer, scheduler, ckpt_path, test_path, batch, device):
    test_path = os.path.join(test_path, 'test')

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

    # for i in range(len(test_images)):
    for i in range(2):
        pred = model(test_images[i])
        predictions.append(pred)
        bar.update(i)
    
    predictions = partition(predictions, batch)

    print("=====> START SAVING OUTPUTS")
    bar = progressbar.ProgressBar(0, num_test_file)

    if not os.path.exists('./output/test/images'):
        os.makedirs('./output/test/images')

    f = open(os.path.join("./output/test", 'test_boxes.txt'), 'w')

    for i in range(len(predictions)):
        # visualize
        for j in range(batch):
            image = cv2.imread(test_files[i][j])
            # 会输出多个box
            bbox = predictions[i][j]['boxes']
            image = visualize_box(image, bbox)
            # plt.imshow()
            # plt.show()
            file_name = test_files[i][j].split('\n')
            file_name = file_name[0].split('/')[-1]
            cv2.imwrite(os.path.join("./output/test/images", file_name), image)

            boxes_text = f"{bbox.shape[0]}: "
            for i in range(bbox.shape[0]):
                x_min, y_min, x_max, y_max = bbox[i].numpy().reshape((4))
                boxes_text += f"[{x_min},{y_min},{x_max},{y_max}], "
            f.write(boxes_text)

            bar.update(i*batch + j)

    print("===== END TEST =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tiny imagenet training')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--use_cbam', type=bool, default=True, help='use cbam or not')
    parser.add_argument('--data', type=str, default='.', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--prefix', type=str, default='test', help='prefix for logging & checkpoint saving')
    parser.add_argument('--ngpu', type=int, default=8, help='numbers of gpu to use')

    parser.add_argument('--ckpt_path', type=str, default='.', help='checkpoint path')
    parser.add_argument('--test_path', type=str, default='./tiny-imagenet-200', help='test directory path')
    parser.add_argument('--test_batch', type=int, default=100, help='number of images to test per batch')

    args = parser.parse_args()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # if args.local_rank != -1:
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl", init_method='env://')

    backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200+1, use_cbam=True)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=5)
    model = FasterRCNN(backbone=backbone, num_classes=200+1, min_size=64, max_size=64)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.1,
        weight_decay=1e-4
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)

    cudnn.benchmark = True

    test(model, optimizer, scheduler, args.ckpt_path, args.test_path, args.test_batch, device)

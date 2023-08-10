import argparse
import os
import random
import shutil

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor

from src.resnet import *
from src.dataset import TinyImagenetDataset
from src.loss import CIoULoss
from train import *
from evaluate import *


def main(args):
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # model = resnet18(args.use_cbam)
    backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200, use_cbam=args.use_cbam)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=5)
    model = FasterRCNN(backbone=backbone, num_classes=200+1)

    # define loss function
    # criterion = nn.SmoothL1Loss().cuda()

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    # model = model.cuda()

    # 程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的
    cudnn.benchmark = True

    # data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        TinyImagenetDataset(
            os.path.join(args.data, 'train'),
            transforms.Compose([
                # transforms.RandomResizedCrop(32),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        TinyImagenetDataset(
            os.path.join(args.data, 'val'),
            transforms.Compose([
                # transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    # evaluate once
    if 1:
        model.eval()
        evaluate(val_loader, model)
        return

    best_acc = 0
    best_model_state = dict()

    # training
    model.train()

    for epoch in range(1, args.epochs+1):
        accuracy = AverageMeter()
        losses = AverageMeter()

        # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        scheduler.step()

        train(train_loader, model, optimizer, (accuracy, losses, epoch))

        if epoch % args.eval == 0:
            acc = evaluate(val_loader, model)
            # remember best model and save checkpoint
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            filename='./checkpoints/%s_checkpoint.pth.tar'%args.prefix
            torch.save(state, filename)
            if is_best:
                shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar'%args.prefix)

    # test
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            os.path.join(args.data, 'test'),
            transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # test
    model.eval()
    evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tiny imagenet training')
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--use_cbam', type=bool, default=True, help='use cbam or not')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight-decay')
    parser.add_argument('--data', type=str, default='.', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--prefix', type=str, default='test', help='prefix for logging & checkpoint saving')
    parser.add_argument('--ngpu', type=int, default=8, help='numbers of gpu to use')
    parser.add_argument('--eval', type=int, default=5, help='numbers of epochs to eval model during training')

    args = parser.parse_args()
    main(args)

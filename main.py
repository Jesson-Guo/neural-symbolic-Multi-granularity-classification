import argparse
import os
import random
import time
import shutil

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .resnet import *
from .train import *
from .evaluate import *


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    model = resnet18(args.use_cbam)
    
    # define loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    # model = model.cuda()

    best_prec1 = 0

    # resume checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的
    cudnn.benchmark = True

    # data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            os.path.join(args.data, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
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
        datasets.ImageFolder(
            os.path.join(args.data, 'val'),
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
    # evaluate once
    if args.evaluate:
        evaluate(val_loader, model, criterion, 0)
        return

    # training
    for epoch in range(1, args.epochs+1):
        # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train(train_loader, model, criterion, optimizer, epoch)

        prec1 = evaluate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
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
    evaluate()

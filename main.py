import argparse
import os
import random
import shutil
import copy

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops import MultiScaleRoIAlign, misc

from src.resnet import *
from src.dataset import TinyImagenetDataset
from train import *
from evaluate import *


def main(args):
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # device = torch.device("cpu")

    # returned layers和anchor_sizes的大小需要相互对应
    num_classes = 200 + 1
    norm_layer = misc.FrozenBatchNorm2d
    trainable_layers = 5
    returned_layers = [1, 2, 3, 4]
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    backbone = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        norm_layer=norm_layer,
        use_cbam=args.use_cbam
    )
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_layers=trainable_layers,
        returned_layers=returned_layers
    )

    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
        canonical_scale=64
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=64,
        max_size=64
    )

    # define loss function
    # criterion = nn.SmoothL1Loss().cuda()

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)
    if args.ngpu > 1:
        # model = nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        # RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
        # This error indicates that your module has parameters that were not used in producing loss. 
        # You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, 
        # and by making sure all `forward` function outputs participate in calculating loss.
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # find_unused_parameters=True
        )

    # 程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的
    cudnn.benchmark = True

    # data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = TinyImagenetDataset(
        os.path.join(args.data, 'train'),
        transforms.Compose([
            # transforms.RandomResizedCrop(32),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_sampler = None
    if args.ngpu > 1:
        train_sampler = DistributedSampler(train_dataset)
    print(train_sampler)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True
    )

    val_dataset = TinyImagenetDataset(
        os.path.join(args.data, 'val'),
        transforms.Compose([
            # transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_sampler = None
    if args.ngpu > 1:
        val_sampler = DistributedSampler(train_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    # evaluate once
    if 1:
        # model.eval()
        evaluate(val_loader, model, device)
        return

    best_acc = 0

    # training
    model.train()

    for epoch in range(1, args.epochs+1):
        losses = AverageMeter()

        train_sampler.set_epoch(epoch)
        train(train_loader, model, optimizer, (losses, epoch), device)

        # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        scheduler.step()

        if epoch % args.eval == 0:
            val_sampler.set_epoch(epoch)
            acc = evaluate(val_loader, model, device)
            # remember best model and save checkpoint
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            state = {
                'epoch': epoch,
                'model': copy.deepcopy(model.module.state_dict()),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            filename='./checkpoints/%s_checkpoint.pth'%args.prefix
            torch.save(state, filename)
            if is_best:
                shutil.copyfile(filename, './checkpoints/%s_model_best.pth'%args.prefix)

    print("***** TRAINING OVER *****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tiny imagenet training')
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
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
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()
    main(args)

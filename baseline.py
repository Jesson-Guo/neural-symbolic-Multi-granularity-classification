import argparse
import os
import random
import shutil

import torch
import torch.distributed
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torchvision.models.resnet import resnet18

from utils.globals import *
from utils.conf import is_main_process
from src.resnet import *
from src.dataset import TinyImagenet200, Imagenet1000, create_dataloader
from src.hierarchy import *
from src.loss import PsychoCrossEntropy
from train import *
from evaluate import *


def main(args):
    init()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    args.start_epoch = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # data loading
    train_dataset = None
    val_dataset = None
    if args.arch == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )
    elif args.arch == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.data_path,
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=args.data_path,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )
    elif args.arch == "tiny-imagenet":
        train_dataset = TinyImagenet200(
            root=args.data_path,
            image_size=64,
            transform=transforms.Compose([
                transforms.RandomCrop(64, 8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
            ]),
            train=True
        )
        val_dataset = TinyImagenet200(
            root=args.data_path,
            image_size=64,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
            ]),
            train=False
        )
    elif args.arch == "imagenet":
        train_dataset = Imagenet1000(
            root=args.data_path,
            image_size=224,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]),
            train=True
        )
        val_dataset = Imagenet1000(
            root=args.data_path,
            image_size=224,
            transform=transforms.Compose([
                transforms.Resize(224 + 32),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]),
            train=False
        )

    train_sampler = None
    val_sampler = None
    if args.ngpu > 1:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)

    train_loader = create_dataloader(args, train_dataset, train_sampler, training=True)
    val_loader = create_dataloader(args, val_dataset, val_sampler, training=False)

    label2id, id2label, lpaths, tree, node_dict = init_global(args, train_dataset.class_to_idx)
    set_value('label2id', label2id)
    set_value('id2label', id2label)
    set_value('lpaths', lpaths)
    set_value('tree', tree)
    set_value('node_dict', node_dict)

    # use resnet18
    if args.model == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=args.num_classes+1, arch=args.arch)
    else:
        # model = resnet18(num_classes=args.num_classes)
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes+1, arch=args.arch)

    if args.resume:
        data = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(data['model'])

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # 程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的
    cudnn.benchmark = True

    if args.resume:
        optimizer.load_state_dict(data['optimizer'])
        scheduler.load_state_dict(data['scheduler'])
        args.start_epoch = data['epoch'] + 1

    if args.ngpu > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )

    best_acc = 0

    # training
    train_losses = torch.zeros((args.epochs+1))
    train_accs = torch.zeros((args.epochs+1, 2))
    eval_losses = torch.zeros((args.epochs+1))
    eval_accs = torch.zeros((args.epochs+1, 2))
    data_len = len(train_loader.dataset)
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler != None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = AverageMeter()
        train_acc = torch.zeros(2).to(device)

        start = time.time()
        if is_main_process():
            train_loader = tqdm.tqdm(train_loader)

        for i, (x, targets) in enumerate(train_loader):
            x = torch.autograd.Variable(x)
            x = x.to(device)

            labels = torch.autograd.Variable(targets)
            labels = labels.to(device)

            out = model(x)
            loss = criterion(out, labels)

            acc1, acc2 = accuracy(out, labels, topk=(1, 2))
            train_acc[0] += acc1
            train_acc[1] += acc2

            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, average=True)
            train_loss.update(reduced_loss.item(), x.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end = time.time()
            if is_main_process():
                train_loader.desc = f'\
                    Epoch: [{epoch}][{i+1}/{len(train_loader)}]\t \
                    Time: {end-start}\t \
                    Loss: {reduced_loss.item()}\t'
            start = end

        torch.distributed.barrier()
        train_acc = reduce_mean(train_acc, average=False)
        train_acc = train_acc / data_len

        if is_main_process():
            print(f'\
                train top1: {train_acc[0].item()}\t \
                train top2: {train_acc[1].item()}')

        train_losses[epoch] = train_loss.avg
        train_accs[epoch] = train_acc

        scheduler.step()

        if epoch % args.eval == 0:
            if val_sampler != None:
                val_sampler.set_epoch(epoch)
            model.eval()
            eval_loss = AverageMeter()
            eval_acc = torch.zeros(2).to(device)

            if is_main_process():
                bar = progressbar.ProgressBar(0, len(val_loader))
            start = time.time()

            with torch.no_grad():
                for i, (x, targets) in enumerate(val_loader):
                    x = x.to(device)
                    labels = targets.to(device)

                    out = model(x)
                    loss = criterion(out, labels)

                    acc1, acc2 = accuracy(out, labels, topk=(1, 2))
                    eval_acc[0] += acc1
                    eval_acc[1] += acc2

                    torch.distributed.barrier()
                    reduced_loss = reduce_mean(loss)
                    eval_loss.update(reduced_loss.item(), x.shape[0])

                    if is_main_process():
                        bar.update(i+1)

            torch.distributed.barrier()
            end = time.time()
            eval_acc = reduce_mean(eval_acc, average=False)
            eval_acc = eval_acc / len(val_loader.dataset)
            if is_main_process():
                print(f'\
                    Epoch: [{epoch}]\t \
                    eval top1: {eval_acc[0].item()}\t \
                    eval top2: {eval_acc[1].item()}\t \
                    time: {end - start}')

            eval_losses[epoch] = eval_loss.avg
            eval_accs[epoch] = eval_acc
            # remember best model and save checkpoint
            is_best = eval_acc[0].item() > best_acc
            best_acc = max(eval_acc[0].item(), best_acc)

            state = {
                'epoch': epoch,
                'tree': get_value('tree'),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }

            if args.ngpu > 1:
                state['model'] = model.module.state_dict()
            else:
                state['model'] = model.state_dict()

            if is_main_process():
                filename=f'./checkpoints/{args.prefix}_checkpoint_{args.local_rank}.pt'
                torch.save(state, filename)
                if is_best:
                    shutil.copyfile(filename, f'./checkpoints/{args.prefix}_model_best_{args.local_rank}.pt')

    status = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'eval_loss': eval_losses,
        'eval_acc': eval_accs,
    }
    torch.save(status, f'./checkpoints/{args.prefix}_status.pt')

    print("***** TRAINING OVER *****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tiny imagenet training')
    parser.add_argument('--seed', type=int, default=72, help='random seed')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--resume', type=int, default=0, help='number of epochs to train')
    parser.add_argument('--use_cbam', type=bool, default=True, help='use cbam or not')
    parser.add_argument('--lamb', type=float, default=1e-3, help='coefficient of the regularization term')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--dqn_lr', type=float, default=1e-4, help='initial dqn learning rate')
    parser.add_argument('--env_decay', type=float, default=0.99, help='initial dqn learning rate')
    parser.add_argument('--cap', type=int, default=100000, help='initial memory capacity')
    parser.add_argument('--eps', type=float, default=0.005, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight-decay')
    parser.add_argument('--data_path', type=str, default='.', help='dataset path')
    parser.add_argument('--arch', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--model', type=str, default='resnet18', help='dataset name')
    parser.add_argument('--agent', type=str, default='sac-d', help='rl agent name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_classes', type=int, default=200, help='num of classes of dataset')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--prefix', type=str, default='test', help='prefix for logging & checkpoint saving')
    parser.add_argument('--ngpu', type=int, default=8, help='numbers of gpu to use')
    parser.add_argument('--eval', type=int, default=1, help='numbers of epochs to eval model during training')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--ckpt', type=str, default='', help='wnids file path')
    parser.add_argument('--wnids', type=str, default='', help='wnids file path')
    parser.add_argument('--words', type=str, default='', help='words file path')
    parser.add_argument('--hier', type=str, default='./structure_released.xml', help='wordnet structure')
    parser.add_argument('--conf', type=float, default=0.4, help='confidence to accept the predicted label')

    args = parser.parse_args()
    main(args)

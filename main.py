import argparse
import os
import random
import shutil

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

from utils.globals import *
from utils.conf import is_main_process
from src.resnet import *
from src.dataset import TinyImagenet200, Imagenet1000, create_dataloader
from src.hierarchy import *
from src.tree import InferTree
from src.loss import PsychoCrossEntropy
from rl.env import TreeClassifyEnv
from rl.agents.DQN import DQN
from rl.agents.sacd import SACD
from train import *
from evaluate import *


def main(args):
    init()
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # device = torch.device("cpu")
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
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=args.num_classes+1, arch=args.arch, use_cbam=args.use_cbam)
        args.dim = 2048
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes+1, arch=args.arch, use_cbam=args.use_cbam)
        args.dim = 512

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
    criterion = PsychoCrossEntropy(args.num_classes)
    # criterion = nn.CrossEntropyLoss()

    model.to(device)

    # 程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的
    cudnn.benchmark = True

    infer_tree = InferTree(args.num_classes, args.dim, nn.CrossEntropyLoss(), args.lamb, device, args.ckpt)

    env = TreeClassifyEnv(args.dim*2, args.env_decay)
    agent = DQN(
        arch=args.arch,
        batch_size=256,
        num_states=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        memory_capacity=args.cap,
        episilon=args.eps,
        lr=args.dqn_lr,
        gamma=args.gamma,
        update_freq=100,
        device=device
    )
    if args.agent == 'sac-d':
        agent = SACD(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            capacity=args.cap,
            device=device
        )

    start_epoch = 1
    losses = AverageMeter()
    if args.resume:
        infer_tree.load_state_dict(data['inference'])
        agent.load_state_dict(data['agent'])
        optimizer.load_state_dict(data['optimizer'])
        scheduler.load_state_dict(data['scheduler'])
        start_epoch = data['epoch'] + 1
        losses = data['losses']

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
        # infer_tree = nn.parallel.DistributedDataParallel(
        #     infer_tree,
        #     device_ids=[args.local_rank],
        #     output_device=args.local_rank,
        #     # find_unused_parameters=True
        # )
        agent = nn.parallel.DistributedDataParallel(
            agent,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # find_unused_parameters=True
        )

    best_acc = 0

    # training
    for epoch in range(start_epoch, args.epochs+1):
        if train_sampler != None:
            train_sampler.set_epoch(epoch)
        train_one_epoch(train_loader, model, infer_tree, env, agent, optimizer, criterion, (losses, epoch), device)
        scheduler.step()

        # train_rl(train_loader, model, infer_tree, env, agent, epoch, device)

        # Sets the learning rate to the initial LR decayed by 10 every 30 epochs

        if epoch % args.eval == 0:
            if val_sampler != None:
                val_sampler.set_epoch(epoch)
            eval_time = time.time()
            acc, rl_acc = evaluate(val_loader, model, env, agent, infer_tree, epoch, device)
            eval_time = time.time() - eval_time
            print(f'\
                Epoch: [{epoch}][{args.epochs+1}]\t \
                Loss: {losses.val}\t \
                acc: {acc}\t \
                rl acc: {rl_acc}\t \
                time: {eval_time}')
            # remember best model and save checkpoint
            is_best = rl_acc > best_acc
            best_acc = max(rl_acc, best_acc)
            state = {
                'epoch': epoch,
                'losses': losses,
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'inference': infer_tree.state_dict()
            }

            if args.ngpu > 1:
                state['model'] = model.module.state_dict()
                state['agent'] = agent.module.state_dict()
            else:
                state['model'] = model.state_dict()
                state['agent'] = agent.state_dict()

            filename=f'./checkpoints/{args.prefix}_checkpoint_{args.local_rank}.pt'
            # if is_main_process():
            torch.save(state, filename)
            if is_best:
                shutil.copyfile(filename, f'./checkpoints/{args.prefix}_model_best_{args.local_rank}.pt')

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

#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 \
        main.py \
        --arch cifar10 \
        --resume 0 \
        --data_path ../data \
        --epochs 200 \
        --batch_size 64 \
        --num_classes 10 \
        --lr 0.0001 \
        --lamb 0.001 \
        --use_cbam 0 \
        --ngpu 2 \
        --prefix RESNET18_CIFAR10 \
        --eval 1 \
        --wnids ../data/cifar-10-batches-py/wnids.txt \
        --words ../data/cifar-10-batches-py/words.txt \
        --hier ./structure_released.xml \
        --cap 800000 \
        --agent dqn \
        --ckpt ./checkpoints/RESNET18_CIFAR10_model_best_0.pt \
        -j 8

python -m torch.distributed.launch --nproc_per_node=4 --master_port 29502 \
        main.py \
        --arch cifar100 \
        --resume 0 \
        --data_path ../data \
        --epochs 200 \
        --batch_size 64 \
        --num_classes 100 \
        --lr 0.0001 \
        --lamb 0.001 \
        --use_cbam 0 \
        --ngpu 2 \
        --prefix RESNET18_CIFAR100 \
        --eval 1 \
        --wnids ../data/cifar-100-python/wnids.txt \
        --words ../data/cifar-100-python/words.txt \
        --hier ./structure_released.xml \
        --cap 800000 \
        --agent dqn \
        --ckpt ./checkpoints/RESNET18_CIFAR100_model_best_0.pt \
        -j 8

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
        main.py \
        --arch tiny-imagenet \
        --resume 0 \
        --data_path ../data \
        --epochs 200 \
        --batch_size 64 \
        --num_classes 200 \
        --lr 0.0001 \
        --lamb 0.001 \
        --use_cbam 0 \
        --ngpu 2 \
        --prefix RESNET18_TINY_IMAGENET \
        --eval 1 \
        --wnids ../data/tiny-imagenet-200/wnids.txt \
        --words ../data/tiny-imagenet-200/words.txt \
        --hier ./structure_released.xml \
        --cap 600000 \
        --agent dqn \
        --ckpt ./checkpoints/RESNET18_CIFAR100_model_best_0.pt \
        -j 8

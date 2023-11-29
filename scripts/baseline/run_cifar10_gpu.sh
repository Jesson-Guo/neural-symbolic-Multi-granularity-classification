#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 \
        baseline.py \
        --arch cifar10 \
        --resume 0 \
        --data_path ../data \
        --epochs 100 \
        --batch_size 64 \
        --num_classes 10 \
        --lr 0.001 \
        --lamb 0.001 \
        --use_cbam 0 \
        --ngpu 2 \
        --prefix RESNET18_CIFAR10_BASELINE \
        --eval 1 \
        --wnids ../data/cifar-10-batches-py/wnids.txt \
        --words ../data/cifar-10-batches-py/words.txt \
        --hier ./structure_released.xml \
        --cap 800000 \
        --agent dqn \
        --ckpt ./checkpoints/RESNET18_CIFAR10_BASELINE_checkpoint_0.pt \
        -j 8

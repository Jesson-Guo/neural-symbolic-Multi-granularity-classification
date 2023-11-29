#!/bin/bash
python main.py \
        --arch cifar10 \
        --data_path ../data \
        --epochs 300 \
        --batch_size 64 \
        --num_classes 10 \
        --lr 0.001 \
        --lamb 0.001 \
        --use_cbam 0 \
        --ngpu 1 \
        --prefix RESNET18_CIFAR10 \
        --eval 1 \
        --wnids ../data/cifar-10-batches-py/wnids.txt \
        --words ../data/cifar-10-batches-py/words.txt \
        --hier ./structure_released.xml \
        --cap 200000 \
        --ckpt ./checkpoints/RESNET18_CIFAR10_model_best.pt \
        -j 4

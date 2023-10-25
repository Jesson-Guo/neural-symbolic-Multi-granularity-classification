#!/bin/bash
python main.py \
        --arch cifar100 \
        --data_path ../data \
        --epochs 300 \
        --batch_size 64 \
        --num_classes 100 \
        --lr 0.001 \
        --lamb 0.001 \
        --use_cbam 0 \
        --ngpu 1 \
        --prefix RESNET18_CIFAR100 \
        --eval 1 \
        --wnids ../data/cifar-100-python/wnids.txt \
        --words ../data/cifar-100-python/words.txt \
        --hier ./structure_released.xml \
        -j 4

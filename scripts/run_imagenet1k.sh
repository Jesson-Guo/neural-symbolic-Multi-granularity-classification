#!/bin/bash
python main.py \
        --arch imagenet \
        --model resnet50 \
        --data_path /root/autodl-tmp \
        --epochs 300 \
        --batch_size 64 \
        --num_classes 1000 \
        --lr 0.001 \
        --lamb 0.001 \
        --use_cbam 0 \
        --ngpu 1 \
        --prefix RESNET50_IMAGENET1K \
        --eval 1 \
        --hier ./structure_released.xml \
        -j 4

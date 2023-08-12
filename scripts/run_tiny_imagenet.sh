#!/bin/bash
python main.py \
        --data ./tiny-imagenet-200 \
        --epochs 50 \
        --batch_size 32 \
        --use_cbam 1 \
        --prefix RESNET18_TINY_IMAGENET_CBAM \
        -j 4

#!/bin/bash
python main_v2.py \
        --arch tiny-imagenet \
        --data_path ../data \
        --epochs 300 \
        --batch_size 64 \
        --num_classes 200 \
        --lr 0.001 \
        --lamb 0.001 \
        --use_cbam 0 \
        --ngpu 1 \
        --prefix RESNET18_TINY_IMAGENET \
        --eval 1 \
        --hier ./structure_released.xml \
        --wnids ../data/tiny-imagenet-200/wnids.txt \
        --words ../data/tiny-imagenet-200/words.txt \
        -j 4

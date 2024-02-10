#!/bin/bash
python main.py \
        --epochs 30 \
        --lr 0.0005 \
        --method tot \
        --k 2 \
        --data cifar10 \
        --words /mnt/data/ztl/mycode/data/cifar-10-batches-py/words.txt \
        --config ./src/vpt/configs/files/simple/cifar10.yaml \
        --tree ./tots/cifar10-2.json \
        --train \
        --naive \
        --pretrained \
        NUM_GPUS 1
        MODEL.MODEL_NAME: "tot_cifar10-2_best.pth"

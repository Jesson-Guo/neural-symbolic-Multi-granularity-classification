#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=3 --master_port 29502 \
        main.py \
        --epochs 10 \
        --lr 0.005 \
        --root /root/autodl-tmp/data \
        --method tot \
        --k 2 \
        --data cifar10 \
        --words /root/autodl-tmp/data/cifar-10-batches-py/words.txt \
        --train \
        --use_timm \
        MODEL.MODEL_NAME ./weights/vit_base_patch16_224_in21k_ft_cifar10.pth

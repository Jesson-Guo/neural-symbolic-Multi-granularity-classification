#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=3 --master_port 29501 \
        main.py \
        --epochs 50 \
        --lr 0.5 \
        --root /root/autodl-tmp/data \
        --method tot \
        --k 10 \
        --data cifar100 \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt \
        --train \
        --use_timm \
        MODEL.MODEL_NAME ./weights/vit_base_patch16_224_in21k_ft_cifar100.pth

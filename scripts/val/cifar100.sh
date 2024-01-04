#!/bin/bash
python main.py \
        --model vit_base_patch16_224.orig_in21k_ft_in1k \
        --ckpt ./weights/vit_base_patch16_224_in21k_ft_cifar100.pth \
        --root /root/autodl-tmp/data \
        --save ./thought.json \
        --load ./cifar100.json \
        --data cifar100 \
        --classes 100 \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt \
        --pretrained

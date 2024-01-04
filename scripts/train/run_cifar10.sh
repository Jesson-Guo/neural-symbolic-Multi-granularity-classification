#!/bin/bash
python main.py \
        --model vit_base_patch16_224.orig_in21k_ft_in1k \
        --ckpt ./weights/vit_base_patch16_224_in21k_ft_cifar100.pth \
        --root /root/autodl-tmp/data \
        --save ./thought.json \
        --load ./cifar10.json \
        --data cifar10 \
        --classes 10 \
        --words /root/autodl-tmp/data/cifar-100-python/words.txt \
        --pretrained
        --config-file=./src/vpt/configs/files/prompt/cifar10.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        MODEL.PROMPT.NUM_TOKENS "10" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        MODEL.MODEL_ROOT "./weights" \
        OUTPUT_DIR "./output"

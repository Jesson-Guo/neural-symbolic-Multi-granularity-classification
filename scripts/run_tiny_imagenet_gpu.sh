 #!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 \
        main.py \
        --data ./tiny-imagenet-200 \
        --epochs 10 \
        --batch_size 64 \
        --use_cbam 1 \
        --ngpu 4 \
        --prefix RESNET50_FPN_TINY_IMAGENET_CBAM \
        -j 8

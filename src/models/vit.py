import timm
import torch
import torch.nn as nn


class ViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, cfg, load_pretrain=True, vis=False):
        model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=cfg.DATA.NUMBER_CLASSES, in_chans=3, img_size=(224, 224))
        super(ViT, self).__init__(**model_args)

    def freeze(self):
        for k, p in self.named_parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x, pre_logits=True)
        out = self.head(x)
        coarse_out = self.head_coarse(x)
        return out, coarse_out

"""
ViT-related models
Note: models return logits instead of prob
"""
import os
import numpy as np
import torch.nn as nn

from src.vpt.models.vit_prompt.vit import PromptedVisionTransformer
from src.vpt.models.mlp import MLP


MODEL_ZOO = {
    "swint_imagenet": "swin_tiny_patch4_window7_224.pth",
    "swint_imagenet_ssl": "moby_swin_t_300ep_pretrained.pth",
    "swins_imagenet": "swin_small_patch4_window7_224.pth",
    "swinb_imagenet_224": "swin_base_patch4_window7_224.pth",
    "swinb_imagenet_384": "swin_base_patch4_window12_384.pth",
    "swinb_imagenet22k_224":  "swin_base_patch4_window7_224_22k.pth",
    "swinb_imagenet22k_384": "swin_base_patch4_window12_384_22k.pth",
    "swinl_imagenet22k_224": "swin_large_patch4_window7_224_22k.pth",
    "sup_vitb8": "ViT-B_8.npz",
    "sup_vitb16_224": "ViT-B_16-224.npz",
    "sup_vitb16": "ViT-B_16.npz",
    "sup_vitl16_224": "ViT-L_16-224.npz",
    "sup_vitl16": "ViT-L_16.npz",
    "sup_vitb8_imagenet21k": "imagenet21k_ViT-B_8.npz",
    "sup_vitb32_imagenet21k": "imagenet21k_ViT-B_32.npz",
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "sup_vitl16_imagenet21k": "imagenet21k_ViT-L_16.npz",
    "sup_vitl32_imagenet21k": "imagenet21k_ViT-L_32.npz",
    "sup_vith14_imagenet21k": "imagenet21k_ViT-H_14.npz",
    "mae_vith14": "mae_pretrain_vit_huge.pth",
    "mae_vitb16": "mae_pretrain_vit_base.pth",
    "mae_vitl16": "mae_pretrain_vit_large.pth",
}


m2featdim = {
    "sup_vitb16_224": 768,
    "sup_vitb16": 768,
    "sup_vitl16_224": 1024,
    "sup_vitl16": 1024,
    "sup_vitb8_imagenet21k": 768,
    "sup_vitb16_imagenet21k": 768,
    "sup_vitb32_imagenet21k": 768,
    "sup_vitl16_imagenet21k": 1024,
    "sup_vitl32_imagenet21k": 1024,
    "sup_vith14_imagenet21k": 1280,
}


class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        self.froze_enc = False

        self.enc = PromptedVisionTransformer(
            cfg.MODEL.PROMPT, cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, num_classes=-1, vis=vis
        )

        if load_pretrain:
            self.enc.load_from(np.load(os.path.join(cfg.MODEL.MODEL_ROOT, MODEL_ZOO[cfg.DATA.FEATURE])))

        self.feat_dim = m2featdim[cfg.DATA.FEATURE]

        for k, p in self.enc.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False

        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * cfg.MODEL.MLP_NUM + [cfg.DATA.NUMBER_CLASSES],
            special_bias=True
        )

    def forward(self, x, return_feature=False):
        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim

        if return_feature:
            return x, x
        x = self.head(x)

        return x

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x

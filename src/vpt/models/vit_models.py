"""
ViT-related models
Note: models return logits instead of prob
"""
import os
import numpy as np
import torch
import torch.nn as nn

from src.vpt.models.vit_prompt.vit import PromptedVisionTransformer
from src.vpt.models.vit_backbone.vit import VisionTransformer
from src.vpt.models.mlp import MLP


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

        transfer_type = cfg.MODEL.TRANSFER_TYPE

        if "prompt" in transfer_type:
            self.enc = PromptedVisionTransformer(cfg.MODEL.PROMPT, cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, num_classes=-1, vis=vis)
        else:
            self.enc = VisionTransformer(cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, num_classes=-1, vis=vis)

        self.feat_dim = m2featdim[cfg.DATA.FEATURE]

        if transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False
        elif transfer_type == "linear":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        # self.head = MLP(
        #     input_dim=self.feat_dim,
        #     mlp_dims=[self.feat_dim] * cfg.MODEL.MLP_NUM + [cfg.DATA.NUMBER_CLASSES],
        #     special_bias=True
        # )
        self.head_coarse = nn.Linear(self.feat_dim, cfg.DATA.NUMBER_COARSE)
        self.head = nn.Linear(self.feat_dim, cfg.DATA.NUMBER_CLASSES)

        for k, p in self.head.named_parameters():
            p.requires_grad = False

        if load_pretrain:
            weights = np.load(os.path.join(cfg.MODEL.MODEL_ROOT, cfg.MODEL.MODEL_NAME))
            self.enc.load_from(weights)
            self.head.weight.copy_(torch.from_numpy(weights["head/kernel"]).T)

    def forward(self, x, return_feature=False):
        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)
        out = self.head(x)
        if return_feature:
            coarse_out = self.head_coarse(x)
            return out, coarse_out
        return out

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x

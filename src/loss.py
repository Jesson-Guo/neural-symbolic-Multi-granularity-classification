import torch.nn as nn
from torch import Tensor
from torchvision import ops


class CIoULoss(nn.Module):
    def __init__(
        self,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return ops.complete_box_iou_loss(input, target, reduction=self.reduction)

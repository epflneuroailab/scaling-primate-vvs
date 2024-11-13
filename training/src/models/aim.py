from typing import  Any, Tuple

import torch
import torch.nn as nn

from .base import ClassifierBase

class AttentionPoolingClassifierWithPreOut(nn.Module):
    """
    Attention Pooling Classifier with Pre-Out.

    This class is adapted from the AIM library (https://github.com/apple/ml-aim/blob/daa3f7122834ce61d0944041434540ed6180086c/aim/torch/layers.py#L337).
    The change enables capturing the feature map before the final classification layer.

    Args:
        pretrained_head (nn.Module): The pretrained head, which is the AttentionPoolingClassifier.
    """

    def __init__(self, pretrained_head: nn.Module):
        super().__init__()
        self.num_heads = pretrained_head.num_heads
        self.scale = pretrained_head.scale

        self.k = pretrained_head.k
        self.v = pretrained_head.v

        self.cls_token = pretrained_head.cls_token
        self.linear = pretrained_head.linear
        self.bn = pretrained_head.bn
        self.pre_out = nn.Identity() # Added

        self.num_queries = pretrained_head.num_queries

    def forward(self, x: torch.Tensor, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the AttentionPoolingClassifierWithPreOut.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor and the feature map before the final classification layer.
        """
        B, N, C = x.shape

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1)
        
        pre_out = self.pre_out(x_cls) # Added

        out = self.linear(pre_out)
        return out, x_cls
    
    
AIM_MODELS = [
    f"aim_{i}" for i in ["600M", "1B", "3B", "7B"]
]


def create_aim_model(**kwargs):
    arch = kwargs.get("arch", None)
    assert arch in AIM_MODELS, f"Invalid aim model: {arch}"
    model = torch.hub.load("apple/ml-aim", arch)
    model.head = AttentionPoolingClassifierWithPreOut(model.head)
    model = ClassifierBase(model, **kwargs)
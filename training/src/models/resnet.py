from typing import List, Union, Optional, Type, Callable

import numpy as np
import torch.nn as nn
import torchvision

from .base import ClassifierBase

class ResNetFlexFilters(torchvision.models.ResNet):
    """
    ResNet with flexible width and layer sizes.
    Adapted from torchvision.models.resnet.ResNet. https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    def __init__(
        self,
        block: Type[Union[torchvision.models.resnet.BasicBlock, torchvision.models.resnet.Bottleneck]],
        layers: List[int],
        filters: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = filters[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, filters[1], layers[0])
        last_channel_size = filters[1] * block.expansion
        for layer_id in range(1, len(layers)):
            if layers[layer_id] > 0:
                last_channel_size = filters[layer_id + 1] * block.expansion
                layer = self._make_layer(block, filters[layer_id + 1], layers[layer_id], stride=2, dilate=replace_stride_with_dilation[layer_id - 1])
            else:
                layer = nn.Identity()
            setattr(self, f"layer{layer_id + 1}", layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_channel_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, torchvision.models.resnet.Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, torchvision.models.resnet.BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

class ResNetBase(ClassifierBase):
    def __init__(
        self,
        module,
        num_classes=1000,
        train_metrics=None,
        val_metrics=None,
        loss_fn=None,
        **kwargs
    ):
        if num_classes != 1000:
            module.fc = nn.Linear(
                in_features=module.fc.in_features,
                out_features=num_classes,
                bias=module.fc.bias is not None,
            )

        super().__init__(module, num_classes, train_metrics, val_metrics, loss_fn)
        
RESNET_MODELS = [
    f"resnet{i}" for i in [18, 34, 50, 101, 152]
]
        
def create_resnet_model(**kwargs):
    arch = kwargs.get("arch", None)
    assert arch in RESNET_MODELS, f"Invalid ResNet model: {arch}"
    weights = kwargs.get("weights", "DEFAULT")
    model = eval(f"torchvision.models.{arch}")(weights=weights)
    model = ResNetBase(model, **kwargs)
    return model


def create_resnetflex_model(**kwargs):
    layer_config = kwargs.pop("layer_config", None)
    assert layer_config is not None, f"Layer config must be specified for ResNetFlexFilters"
    if "w" in layer_config:
        config = layer_config.split("-")
        assert len(config) in [5, 6], f"Invalid layer config: {layer_config}"
        config_str = config[0]
        config = [int(x) for x in config[1:]]
        if len(config) == 4:
            scale = 2
            geomspace_start, geomspace_end, geomspace_num, geomspace_id = config
        else:
            scale, geomspace_start, geomspace_end, geomspace_num, geomspace_id = config
        layers = [2, 2, 2, 2]
        filters = np.array([64, 64, 128, 256, 512]) * scale
        divisors = np.geomspace(geomspace_start, geomspace_end, geomspace_num)
        divisor = divisors[geomspace_id]
        if "r" in config_str:
            filters = np.round(filters / divisor / 8).astype(int)*8
        else:
            filters = (filters / divisor).astype(int)
    elif "d" in layer_config:
        config = layer_config.split("-")
        assert len(config) == 5, f"Invalid layer config: {layer_config}"
        layers = [int(x) for x in config[1:]]
        filters = np.array([64, 64, 128, 256, 512])
    block = kwargs.pop("block", torchvision.models.resnet.BasicBlock)
    model = ResNetFlexFilters(block, layers, filters, **kwargs)
    model = ResNetBase(model, **kwargs)
    return model



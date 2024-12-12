
import torch.nn as nn
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import torchvision


from .base import ClassifierBase


# EfficientNet pretrained weights hash check is broken
# https://github.com/pytorch/vision/issues/7744
# Temporary fix:
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict


EFFICIENTNET_MODELS = [
    f"efficientnet_b{i}" for i in range(8)
] + [
    f"efficientnet_v2_{i}" for i in ['s', 'm', 'l']
]


class EfficientNetBase(ClassifierBase):
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
            module.classifier[1] = nn.Linear(
                in_features=module.classifier[1].in_features,
                out_features=num_classes,
                bias=module.classifier[1].bias is not None,
            )
        super().__init__(module, num_classes, train_metrics, val_metrics, loss_fn)
        
def create_efficientnet_model(**kwargs):
    arch = kwargs.get("arch", None)
    assert arch in EFFICIENTNET_MODELS, f"Invalid EfficientNet model: {arch}"
    weights = kwargs.get("weights", "DEFAULT")
    model = eval(f"torchvision.models.{arch}")(weights=weights)
    model = EfficientNetBase(model, **kwargs)
    return model

import torch.nn as nn
import torchvision
import timm

from .base import ClassifierBase



class ViTBase(ClassifierBase):
    def __init__(
        self,
        module,
        num_classes=1000,
        train_metrics=None,
        val_metrics=None,
        loss_fn=None,
        **kwargs
    ):
        if num_classes is not None and num_classes>0 and num_classes != 1000:
            module.head = nn.Linear(
                in_features=module.head.in_features,
                out_features=num_classes,
                bias=module.head.bias is not None,
            )
        super().__init__(module, num_classes, train_metrics, val_metrics, loss_fn)


class AlexNetBase(ClassifierBase):
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
            module.classifier[-1] = nn.Linear(
                in_features=module.classifier[-1].in_features,
                out_features=num_classes,
                bias=module.classifier[-1].bias is not None,
            )
        super().__init__(module, num_classes, train_metrics, val_metrics, loss_fn)

        
def create_alexnet_model(**kwargs):
    arch = kwargs.get("arch", None)
    assert arch in [
        "alexnet"
    ], f"Invalid AlexNet model: {arch}"
    weights = kwargs.get("weights", "DEFAULT")
    model = eval(f"torchvision.models.{arch}")(weights=weights)
    model = AlexNetBase(model, **kwargs)
    return model


VIT_MODELS = [
    f"vit_{i}_{j}" 
    for i in ["b", "l"] 
    for j in [16, 32]
] + ["vit_h_14"]


def create_vit_model(**kwargs):
    use_timm = kwargs.get("use_timm", False)
    arch = kwargs.get("arch", None).replace(':', '.')
    if not use_timm:
        weights = kwargs.get("weights", "DEFAULT")
        assert arch in VIT_MODELS, f"Invalid ViT model: {arch}"
        model = eval(f"torchvision.models.{arch}")(weights=weights)
        model = ViTBase(model, **kwargs)
    else:
        timm_config = {
            'model_name': arch, 
            'pretrained': kwargs.get("pretrained", False),
            'drop_path_rate': kwargs.get("drop_path_rate", None),
            'init_values': kwargs.get("init_values", None),
            'dynamic_img_size': kwargs.get("dynamic_img_size", None),
        }
        num_classes = kwargs.get("num_classes", None)
        if num_classes is not None and num_classes != 0:
            timm_config['num_classes'] = num_classes
        model = timm.create_model(**timm_config)
        model = ViTBase(model, **kwargs)
        
    return model
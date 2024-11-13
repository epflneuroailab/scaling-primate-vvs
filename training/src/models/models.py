from collections import OrderedDict
from typing import List, Union, Optional, Type, Callable, Any, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm
import open_clip
import torch.nn.functional as F


from .base import ClassifierBase
from .resnet import create_resnet_model, create_resnetflex_model, RESNET_MODELS
from .efficientnet import create_efficientnet_model, EFFICIENTNET_MODELS
from .cornet import create_cornet_model, CORNET_MODELS
from .others import create_alexnet_model, create_vit_model, VIT_MODELS
from .aim import create_aim_model, AIM_MODELS
from .convnext import create_convnext_model, CONVNEXT_MODELS
from .deit import create_deit_model, DEIT_MODELS
from .ssl import wrap_ssl_model
from .adversarial import wrap_adv_model


def create_model(**kwargs):
    """
    Returns a model based on the specified architecture.

    Args:
        arch (str): The architecture of the model.
        use_timm (bool, optional): Whether to use the timm library for model creation. Defaults to False.
        use_open_clip (bool, optional): Whether to use the open_clip library for model creation. Defaults to False.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to False.
        checkpoint_path (str, optional): Path to a checkpoint to load. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the model constructor.

    Returns:
        composer.models.ComposerModel: The created model.

    Raises:
        ValueError: If the specified architecture is invalid.
    """
    arch = kwargs.get("arch")
    use_timm = kwargs.get("use_timm")
    use_open_clip = kwargs.get("use_open_clip")
    pretrained = kwargs.get("pretrained")
    checkpoint = kwargs.get("checkpoint", None)
    layer_config = kwargs.get("layer_config")
    load_model_ema = kwargs.get("load_model_ema", False)
    ssl_method = kwargs.get("ssl_method")
    adv_config = kwargs.get("adv_config")

    if pretrained:
        pretrained_weights = kwargs.get("pretrained_weights", "DEFAULT")
        weights = pretrained_weights
    else:
        weights = None
    kwargs["weights"] = weights
    
    if not use_timm and not use_open_clip:
        # PyTorch and custom models
        if arch in CORNET_MODELS:
            model = create_cornet_model(**kwargs)
        elif arch in RESNET_MODELS:
            model = create_resnet_model(**kwargs)
        elif arch in EFFICIENTNET_MODELS:
            model = create_efficientnet_model(**kwargs)
        elif arch in VIT_MODELS:
            model = create_vit_model(**kwargs)
        elif arch in CONVNEXT_MODELS:
            model = create_convnext_model(**kwargs)
        elif arch in DEIT_MODELS:
            model = create_deit_model(**kwargs)
        elif arch in AIM_MODELS:
            model = create_aim_model(**kwargs)
        elif "resnetflex" in arch:
            model = create_resnetflex_model(**kwargs)
        elif arch == "alexnet":
            model = create_alexnet_model(**kwargs)
        else:
            raise ValueError(f"Invalid model: {arch}")

    elif use_timm:
        if "vit" in arch:
            # ViT models from timm
            model = create_vit_model(**kwargs)
        else:
            model = timm.create_model(arch, pretrained=pretrained, num_classes=kwargs["num_classes"])
            model = ClassifierBase(model, **kwargs)
    elif use_open_clip:
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=checkpoint)
        model = model.visual
        model.preprocess = preprocess
        model = ClassifierBase(model, **kwargs)
    else:
        raise ValueError(f"Invalid model: {arch}")
    
    if ssl_method:
        model = wrap_ssl_model(model, **kwargs)
    elif adv_config:
        model = wrap_adv_model(model, **kwargs)
    
    if checkpoint:
        if 'http' in checkpoint:
            model_id = kwargs.get("run_name", arch)
            epoch = checkpoint.split("ep")[-1].split(".")[0]
            state_dict = torch.hub.load_state_dict_from_url(
                checkpoint,
                check_hash=True,
                file_name=f"{model_id}_ep{epoch}.pt",
                map_location="cpu",
            )
            kwargs['checkpoint'] = None
        else:
            state_dict = torch.load(checkpoint, map_location="cpu")
        if load_model_ema:
            model.load_state_dict(state_dict["state"]["model_ema_state_dict"], strict=True)
            print(f"Loaded EMA model from checkpoint: {checkpoint} to model.model_ema")
        else:
            model.load_state_dict(state_dict["state"]["model"], strict=True)
            print(f"Loaded model from checkpoint: {checkpoint}")

    if kwargs.get("sync_bn", False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model

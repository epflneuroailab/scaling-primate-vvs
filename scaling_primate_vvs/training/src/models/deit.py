# Adaptepted from https://github.com/facebookresearch/ConvNeXt

import json
from functools import partial

import numpy as np
import torchvision
from timm.models import create_model
from timm.layers.drop import DropPath
from timm.models.registry import register_model
from timm.models.deit import _create_deit
from timm.models.vision_transformer import VisionTransformer

from .base import ClassifierBase


def replace_drop_path(module, drop_path_rate=0.1):
    '''
    Recursively replace drop path layers with a fixed rate,
    as in the original DeiTv3 paper. In timm, the drop path rate increases with depth.
    '''
    for attr_name in dir(module):
        target_attr = getattr(module, attr_name)
        if 'drop_path' in attr_name:
            new_layer = DropPath(drop_prob=drop_path_rate)
            setattr(module, attr_name, new_layer)

    for child_name, child_module in module.named_children():
        replace_drop_path(child_module, drop_path_rate)
        
@register_model
def deit3_tiny_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

def create_deitflex_model(
    layer_config: str,
    **kwargs
    ):
    """
    """
    
    def get_n_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def prettify(x:float) -> str:
        if x < 1e3:
            return str(x)
        if x < 1e6:
            return f"{x/1e3:.1f}K"
        if x < 1e9:
            return f"{x/1e6:.1f}M"
        return f"{x/1e9:.1f}B"

    # width, depth, n_heads = 192, 12, 3 # ViT-Tiny
    width, depth, n_heads = 384, 12, 6 # ViT-Small
    
    config = layer_config.split("-")
    config_str = config[0]
    if config_str == "l":
        # Compact format, specs are in config file
        assert len(config) == 2, f"Invalid layer config: {layer_config}"
        scale = 1.0
        geom_id = int(config[1])
        geom_start = kwargs['geomspace_start']
        geom_end = kwargs['geomspace_end']
        geom_num = kwargs['geomspace_num']
        # geom_start = kwargs.get('geomspace_start', 1)
        # geom_end = kwargs.get('geomspace_end', 8)
        # geom_num = kwargs.get('geomspace_num', 4)
    elif config_str == "w":
        assert len(config) in [5, 6], f"Invalid layer config: {layer_config}"
        config = [int(x) for x in config[1:]]
        if len(config) == 4:
            scale = 1.0
            geom_start, geom_end, geom_num, geom_id = config
        else:
            scale, geom_start, geom_end, geom_num, geom_id = config
            
    assert 0 <= geom_id and geom_id < geom_num, f"Invalid geom_id: {geom_id}"
    divisors = scale*np.geomspace(geom_start, geom_end, geom_num+1)
    d = divisors[geom_id+1] # The first divisor is always 1, so we skip it
    width = np.round(width / d / n_heads).astype(int)*n_heads
    
    init_values = float(kwargs["layer_scale_init_value"])
    drop_path_rate = float(kwargs["drop_path_rate"])
    num_classes = int(kwargs["num_classes"])
    
    model_args = dict(
        patch_size=16, 
        embed_dim=width,
        depth=depth, 
        num_heads=n_heads, 
        no_embed_class=True, 
        init_values=init_values,
        drop_path_rate=drop_path_rate,
        num_classes=num_classes
    )
    model = _create_deit('', pretrained=False, **dict(model_args))
    
    n = get_n_params(model)
    print(f"DeiT-Flex: {layer_config} -> {prettify(n)} parameters")
    
    return model

class DeiTBase(ClassifierBase):
    def __init__(
        self,
        module,
        num_classes=1000,
        train_metrics=None,
        val_metrics=None,
        loss_fn=None,
        **kwargs
    ):
        super().__init__(module, num_classes, train_metrics, val_metrics, loss_fn)

DEIT_MODELS = [
    f"deit3_{i}_patch16_224" for i in ["tiny", "small", "medium", "base", "large"]
] + ["deit3_huge_patch14_224"]


def create_deit_model(
    arch:str, 
    pretrained:bool = False, 
    num_classes:int = 1000, 
    drop_path_rate:float = 0.0, 
    layer_scale_init_value: float = 1e-4, 
    **kwargs
    ):
    """
    
    """
    if "deitflex" in arch:
        model = create_deitflex_model(
            num_classes=num_classes, 
            drop_path_rate=float(drop_path_rate),
            layer_scale_init_value=float(layer_scale_init_value),
            **kwargs
        )
    else:
        assert arch in DEIT_MODELS, f"Invalid deit model: {arch}"
        
        model = create_model(
            arch,
            pretrained=pretrained, 
            num_classes=num_classes, 
            drop_path_rate=float(drop_path_rate),
            init_values=float(layer_scale_init_value),
            # **kwargs
        )
        
    replace_drop_path(model, drop_path_rate)
    
    model = DeiTBase(model, **kwargs)
    
    return model

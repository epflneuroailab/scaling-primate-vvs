# Adaptepted from https://github.com/facebookresearch/ConvNeXt

import json
from functools import partial

import numpy as np
import torch
import torchvision
from torchvision.models.convnext import CNBlockConfig, _convnext
# from timm.models import create_model

from .base import ClassifierBase

def get_num_layer_for_convnext(var_name):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three 
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    num_max_layer = 12
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split('.')[1])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    elif var_name.startswith("stages"):
        stage_id = int(var_name.split('.')[1])
        block_id = int(var_name.split('.')[2])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // 3 
        elif stage_id == 3:
            layer_id = 12
        return layer_id
    else:
        return num_max_layer + 1

class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_convnext(var_name)


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


    
def get_convnext_parameters(model, weight_decay, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    """
    Get the parameters and weight decay for the ConvNext model.

    Args:
        model (nn.Module): The ConvNext model.
        weight_decay (float): The weight decay value.
        get_num_layer (function, optional): A function to get the number of layers in the model. Defaults to None.
        get_layer_scale (function, optional): A function to get the scale of each layer in the model. Defaults to None.
        filter_bias_and_bn (bool, optional): Whether to filter out bias and batch normalization parameters. Defaults to True.
        skip_list (dict, optional): A dictionary of parameters to skip weight decay. Defaults to None.

    Returns:
        tuple: A tuple containing the parameters and weight decay value.
    """
    # Implementation code goes here
    pass
    
    if filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()
        
    return {'parameters':parameters, 'weight_decay':weight_decay}

def create_convnextflex_model(
    layer_config: str,
    **kwargs
):
    
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
    
    depth = np.array([3, 3, 9, 3]) # ConvNext-Tiny
    # depth = np.array([3, 3, 27, 3]) # ConvNext-Small
    
    config = layer_config.split("-")
    config_str = config[0]
    if config_str == "l":
        # Compact format
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

    width = np.array([96, 192, 384, 768])
    width = np.round(width / d / 8).astype(int)*8
    
    block_setting = [
        CNBlockConfig(width[0], width[1], depth[0]),
        CNBlockConfig(width[1], width[2], depth[1]),
        CNBlockConfig(width[2], width[3], depth[2]),
        CNBlockConfig(width[3], None, depth[3]),
    ]
    
    stochastic_depth_prob = float(kwargs['drop_path_rate'])
    layer_scale = float(kwargs['layer_scale_init_value'])
    model = _convnext(
        block_setting=block_setting, 
        stochastic_depth_prob=stochastic_depth_prob,
        layer_scale=layer_scale,
        weights=None, 
        progress=False
    )
    n = get_n_params(model)
    print(f"ConvNeXt-Flex: {layer_config} -> {prettify(n)} parameters")
    
    return model

class ConvNeXtBase(ClassifierBase):
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
        
        
CONVNEXT_MODELS = [
    f"convnext_{i}" for i in ["tiny", "small", "base", "large"]
]

def create_convnext_model(
    arch:str, 
    weights: str = None,
    num_classes:int = 1000, 
    drop_path_rate:float = 0.0, 
    layer_scale_init_value: float = 1e-6, 
    head_init_scale: float = 1.0,
    layer_decay: float = 1.0,
    **kwargs
    ):
    """
    Returns a ConvNext model.

    Args:
        arch (str): The architecture of the model.
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Defaults to False.
        num_classes (int, optional): The number of classes for the classification task. Defaults to 1000.
        drop_path_rate (float, optional): The drop path rate for the model. Defaults to 0.0.
        layer_scale_init_value (float, optional): The initial value for layer scale. Defaults to 1e-6.
        head_init_scale (float, optional): The initial scale for the model's head. Defaults to 1.0.
        layer_decay (float, optional): The decay value for the layers. Defaults to 1.0.

    Returns:
        model: The ConvNext model.
    """
    # model = create_model(
    #     arch,
    #     pretrained=pretrained, 
    #     num_classes=num_classes, 
    #     drop_path_rate=float(drop_path_rate),
    #     ls_init_value=float(layer_scale_init_value),
    #     head_init_scale=float(head_init_scale),
    #     # **kwargs
    # )
    # model = torchvision.models.convnext_tiny()
    print(f"Creating ConvNext model: {arch}")
    if 'convnextflex' in arch:
        model = create_convnextflex_model(
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            **kwargs)
    else:
        assert arch in CONVNEXT_MODELS, f"Invalid ConvNeXt model: {arch}"
        model = eval(f"torchvision.models.{arch}")(weights=weights)
    
    if num_classes != 1000:
        model.classifier[2] = torch.nn.Linear(
            in_features=model.classifier[2].in_features, 
            out_features=num_classes, 
            bias=model.classifier[2].bias is not None
        )
    
    if layer_decay < 1.0 or layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    get_parameters = partial(
        get_convnext_parameters, 
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None
    )
    
    # Add get_parameters method to the model
    setattr(model, 'get_parameters', get_parameters)
    
    model = ConvNeXtBase(model, **kwargs)
    
    return model

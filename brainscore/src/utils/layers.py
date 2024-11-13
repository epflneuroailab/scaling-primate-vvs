import pprint

import torch.nn as nn
import torchvision.models as M
import timm.models as TM
import timm.layers as L

def get_layers(model, layer_type='all', sort=False, verbose=False):
    layers = set()
    known_types = {
        'conv': (nn.Conv2d,),
        'fc': (nn.Linear,),
        'bn': (nn.BatchNorm2d,),
        'ln': (nn.LayerNorm,),
        'avgpool': (nn.AdaptiveAvgPool2d,L.SelectAdaptivePool2d),
        'maxpool': (nn.AdaptiveMaxPool2d,L.AdaptiveAvgMaxPool2d, nn.MaxPool2d),
        'attn_proj': (nn.modules.linear.NonDynamicallyQuantizableLinear,),
        'relu': (nn.ReLU,), 
        'silu': (nn.SiLU,), 
        'gelu': (nn.GELU, L.activations.GELU),
        'identity': (nn.Identity,),
        'hardswish': (nn.Hardswish,),
        'gelutanh': (L.activations.GELUTanh,),
        'sequential': (nn.Sequential,),
        'cnblock': (M.convnext.CNBlock,),
        'attn': (TM.vision_transformer.Attention, TM.convit.GPSA),
        'mlp': (L.mlp.Mlp, L.mlp.GlobalResponseNormMlp),
        'grn': (L.mlp.GlobalResponseNorm,),
        'davit': (TM.davit.ConvPosEnc, TM.davit.ChannelBlock, TM.davit.SpatialBlock, TM.davit.WindowAttention, TM.davit.ChannelAttention )
        }
    if layer_type == 'all':
        layers = [name for name, _ in model.named_modules() if len(name) > 0]
        if sort:
            layers = sorted(layers)
        return layers
    elif layer_type == 'all_known':
        layer_types= list(known_types.keys())
    else:
        layer_types = layer_type.strip().split(',')
        
    selected_layers = {}
    for layer_type in layer_types:
        assert layer_type in list(known_types.keys()), \
            f'Layer type {layer_type} not supported, please choose one of {known_types.keys()}'
        layer_type = known_types[layer_type]
        
        # Iterate over all modules in the model
        for name, module in model.named_modules():
            # Check if module matches the type of layer we are looking for
            if any([isinstance(module, lt) for lt in layer_type]):
                if  nn.Linear in layer_type and isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
                    # Skip attention projection layers when considering linear layers
                    continue
                if 'activation' in name:
                    # Temporary fix for efficientnet_b0
                    continue
                if '.head' in name:
                    # Temporary fix for vit_b_16
                    continue
                layers.add(name)
                selected_layers[name] = module
    
    print(f"Model: {model.__class__.__name__}")
    if verbose:
        pprint.pprint(selected_layers)
    else:
        print(f"Selected layers: ", list(selected_layers.keys()))
    # print(f"Selected layers: ", selected_layers)
    layers = list(layers)
    if sort:
        layers = sorted(layers)
    return layers

            
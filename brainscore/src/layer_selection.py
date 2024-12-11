import functools
import argparse
import json
from pathlib import Path
import sys
import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import timm
from brainscore_vision.model_helpers.brain_transformation import (
    RegionLayerMap,
    LayerSelection,
    STANDARD_REGION_BENCHMARKS,
)

if '../' not in sys.path:
    sys.path.append('../')

from training import create_model

from .utils import get_layers, wrap_model, create_argparser, PostprocessWrapper
from .config import BENCHMARKS, MODEL_IDENTIFIERS


os.environ["RESULTCACHING_DISABLE"] = "brainscore.score_model,model_tools"
os.environ["RESULTCACHING_HOME"] = "/work/upschrimpf1/akgokce/.result_caching"

def main(args):
    model = create_model(**vars(args))
    model.eval()
    
    if hasattr(model, "module"):
        model = model.module
    elif hasattr(model, "model"):
        model = model.model
    elif hasattr(args, 'ssl_method') and args.ssl_method is not None:
        model = model
    else:
        raise ValueError("Model not found")

    if args.arch in MODEL_IDENTIFIERS:
        model_identifier = MODEL_IDENTIFIERS[args.arch]
    else:
        model_identifier = args.arch
        
    layers = get_layers(model, layer_type=args.layer_type)
    
        
    print(layers)
    if args.use_timm:
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
    elif args.use_open_clip:
        transforms = model.preprocess
        # OpenCLIP models permute the axes, so we need to undo that
        postprocess_func = lambda x: x.permute(1, 0, 2)  # LND -> NLD
        model = PostprocessWrapper(model, layers, postprocess_func)
        layers = model.layers_
    else:
        transforms = None
        
    activations_model = wrap_model(
        model_identifier, 
        model, 
        transforms=transforms, 
        resize_size=args.resize_size, 
        crop_size=args.crop_size, 
        interpolation=args.interpolation
        )

    layer_selection = LayerSelection(model_identifier, activations_model, layers, visual_degrees=8)
    
    region_layer_map = RegionLayerMap(layer_selection=layer_selection, region_benchmarks=STANDARD_REGION_BENCHMARKS)
    
    region2layer = {}
    for region in tqdm(STANDARD_REGION_BENCHMARKS.keys(), desc="Processing regions"):
        print(f"\nCurrent region: {region}\n")
        region2layer[region] = region_layer_map[region]
        
    results = {
        'config': vars(args), 
        'region2layer': region2layer,
        'layers': layers
    }

    save_path = Path(args.save_dir) / f"{args.run_name}.json"
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=False)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(results)
    print(f"Saved layer selection to {save_path}")


if __name__ == "__main__":
    
    parser = create_argparser()
    parser.set_defaults(save_dir='./layer_selection')
    args = parser.parse_args()
    print(args)
    main(args)

import functools
import argparse
import json
from pathlib import Path
import sys
import os
from tqdm.auto import tqdm
import torch
import timm

from brainscore_vision import load_benchmark
from brainscore_vision.model_helpers.activations.pca import LayerPCA
from brainscore_vision.model_helpers.brain_transformation import (
    LayerScores,
    STANDARD_REGION_BENCHMARKS,
)

if '../' not in sys.path:
    sys.path.append('../')

from training import create_model
from .utils import get_layers, wrap_model, create_argparser
# from .config import BENCHMARKS, MODEL_IDENTIFIERS
from .random_projection import LayerRandomProjection




def main(args):
    
    if args.append_layer_type:
        assert len(args.layer_type.split(",")) == 1, \
        "Only one layer type can be selected when appending layer type to output file name"
        
    assert args.output_method in ['combined', 'separate'], "Invalid output method"
    
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

    scores = {}


        
    model_identifier = f"{args.run_name}_nc-{args.n_compression_components}_{args.layer_type}"
    
    layers = get_layers(model, layer_type=args.layer_type)
    # layers = ['trunk.blocks.2.mlp.act']
    print(layers)
    
    if len(layers) == 0:
        print(f"No layers found for the given layer type {args.layer_type} and model")
        return
    
    if args.use_timm:
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
    elif args.use_open_clip:
        transforms = model.preprocess
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
        
    layer_scoring = LayerScores(model_identifier=model_identifier, activations_model=activations_model, visual_degrees=8)
    
    if not args.disable_pca:
        pca_hooked = LayerPCA.is_hooked(layer_scoring._activations_model)
        if not pca_hooked:
            pca_handle = LayerPCA.hook(layer_scoring._activations_model, n_components=args.n_compression_components)
            
    if args.enable_random_projection:
        random_projection_hooked = LayerRandomProjection.is_hooked(layer_scoring._activations_model)
        if not random_projection_hooked:
            random_projection_handle = LayerRandomProjection.hook(layer_scoring._activations_model, n_components=args.n_compression_components, projection_type='sparse')

    best_scores = {}
    for region, benchmark in tqdm(STANDARD_REGION_BENCHMARKS.items(), desc="Processing regions"):
        print(f"\nCurrent region: {region}\n")
        layer_scores = layer_scoring(benchmark=benchmark, benchmark_identifier=region,
                                                layers=layers, prerun=True)
        layer_scores = {layer: layer_scores.sel(layer=layer).values.tolist() for layer in layers}
        scores[region] = layer_scores
        print(layer_scores)
        # best_layer = sorted(layer_scores.items(), key= lambda x: x[1][0], reverse=True)
        best_layer = sorted(layer_scores.items(), key= lambda x: x[1], reverse=True)
        best_scores[region] = best_layer[0]
        
    if args.output_method == 'combined':
        results = {
            'region2Layer': best_scores,
            'scores': scores,
            'layers': layers,
            'config': vars(args), 
        }

        if args.append_layer_type:
            save_path = Path(args.save_dir) / f"{args.run_name}_{args.layer_type}_layer-scores.json"
        else:
            save_path = Path(args.save_dir) / f"{args.run_name}_layer-scores.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        print(results)
        print(f"Saved layer selection to {save_path}")
    elif args.output_method == 'separate':
        for layer in layers:
            results = {
                'config': vars(args), 
                'scores': {region: {layer: scores[region][layer]} for region in scores},
                'layers': [layer],
            }
            save_path = Path(args.save_dir) / f"{args.run_name}_{layer}_layer-scores.json"
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=False)
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)

            print(results)
            print(f"Saved layer selection to {save_path}")
    else:
        raise ValueError("Invalid output method")

if __name__ == "__main__":
    parser = create_argparser()
    parser.set_defaults(save_dir='./outputs/layer_scoring')
    args = parser.parse_args()
    print(args)
    main(args)
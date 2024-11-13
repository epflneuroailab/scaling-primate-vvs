import functools
import argparse
import json
from pathlib import Path
import sys
import os
import yaml

from tqdm.auto import tqdm
import timm

from brainscore_vision.model_helpers.brain_transformation import ModelCommitment


if '../' not in sys.path:
    sys.path.append('../')

from training import create_model
from .utils import get_layers, wrap_model, create_argparser, postprocess_score
from .config import BENCHMARKS


os.environ["RESULTCACHING_HOME"] = "/work/upschrimpf1/akgokce/.result_caching"


def main(args):
    config = vars(args)
    
    if args.append_layer_type:
        assert len(args.layer_type.split(",")) == 1, \
            "Only one layer type can be selected when appending layer type to output file name"
    
    model = create_model(**config)
    model.eval()
    
    if args.ssl_method is not None and hasattr(model, "backbone"):
        model = model.backbone
    
    if hasattr(model, "module"):
        model = model.module
    elif hasattr(model, "model"):
        model = model.model
    elif hasattr(args, 'ssl_method') and args.ssl_method is not None:
        model = model
    else:
        raise ValueError("Model not found")

    scores = {}
    
    if args.benchmark_layer:
        layers = [args.benchmark_layer]
        model_identifier = args.run_name + "_" + args.benchmark_layer
    else:
        layers = get_layers(model, layer_type=args.layer_type)
        model_identifier = args.run_name + "_" + (args.layer_type).replace(",", "-")
        
    # layers = layers[0:2]
    print(layers)
    
    if args.benchmark_layer:
        layer = args.benchmark_layer
        # save_path = Path(args.save_dir) / args.arch / args.run_name / f"{args.run_name}_{layer}_benchlayers.json"
        save_path = Path(args.save_dir) / args.arch / args.run_name / f"{layer}.json"
    else:
        save_path = Path(args.save_dir) / args.arch / args.run_name / f"{args.run_name}_benchlayers.json"
        
    if save_path.exists() and args.skip_existing:
        print(f"Skipping existing file {save_path}")
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
    
    results = {
        'scores': {},
        'config': config,
        'layers': layers
    }
    for layer in tqdm(layers, desc="Processing layers"):
        print(f"\nCurrent layer: {layer}\n")
        region2layer = {region:layer for region in BENCHMARKS.keys()}
        brain_model = ModelCommitment(
            identifier=model_identifier, 
            activations_model=activations_model, 
            layers=[layer], 
            region_layer_map=region2layer, 
            behavioral_readout_layer=layer
        )

        scores = {}
        for region, benchmark in tqdm(BENCHMARKS.items(), desc="Processing regions"):
            try:
                print(f"\nCurrent region: {region}\n")
                score = benchmark(brain_model)
                score = postprocess_score(score)
                scores[region] = score
                print(score)
            except Exception as e:
                print(f"Error in {region} for {layer}: {e}")
                scores[region] = -1
            
        results['scores'][layer] = scores
        
    if len(results['scores']) > 1:
        best_scores = {region: {'layer': None, 'score': 0} for region in BENCHMARKS.keys()}
        for region in BENCHMARKS.keys():
            region_scores = {layer:v[region]['score'] for layer, v in results['scores'].items()}

            best_layer = max(region_scores, key=region_scores.get)
            best_scores[region]['layer'] = best_layer
            best_scores[region]['score'] = region_scores[best_layer]
        results['best_scores'] = best_scores

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(results)
    print(f"Saved layer benchmarks to {save_path}")

if __name__ == "__main__":
    parser = create_argparser()
    parser.set_defaults(save_dir='./layer_scoring')
    
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open("../" + args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        default_config = cfg.get('defaults', None)
        if default_config is not None:
            # Load a base configuration files if exists
            with open("../" + default_config, 'r') as f:
                defaults = yaml.safe_load(f)
            # Update the defaults with the current configuration
            cfg.update({
                k: v for k, v in defaults.items() if k not in cfg
            })
        parser.set_defaults(**cfg)

    parser.set_defaults(save_dir='./outputs/benchmark_layers')
    args = parser.parse_args(remaining)

    print(args)
    main(args)

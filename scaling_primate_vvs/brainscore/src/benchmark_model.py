import argparse
import json
from pathlib import Path
import sys
import re
import yaml

from tqdm.auto import tqdm
import timm
from composer.utils import reproducibility

from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import load_benchmark

if '../' not in sys.path:
    sys.path.append('../')

from scaling_primate_vvs.training import create_model
from .utils import wrap_model, create_argparser, PostprocessWrapper, postprocess_score
from .config import BENCHMARKS, BENCHMARKS_IDENTIFIERS, MODEL_IDENTIFIERS, MODEL_COMMITMENTS, ABLATION



def brain_wrap_model(identifier, model, model_commitment, transforms, resize_size, crop_size, interpolation):
    
    activations_model = wrap_model(identifier, model, transforms, resize_size, crop_size, interpolation)
    
    # model_commitment = MODEL_COMMITMENTS[identifier][training_dataset]
    layers = model_commitment['layers']
    region2layer = model_commitment['region2layer']
    behavioral_readout_layer = model_commitment['behavioral_readout_layer']
    
    brain_model = ModelCommitment(identifier=identifier, activations_model=activations_model, 
        layers=layers, region_layer_map=region2layer, behavioral_readout_layer=behavioral_readout_layer)
    
    return brain_model




def score_model(
    model_identifier, model, model_commitment, benchmark_identifier, transforms, resize_size, crop_size, interpolation
):

    brain_model = brain_wrap_model(
        identifier=model_identifier, model=model, model_commitment=model_commitment, transforms=transforms, resize_size=resize_size, crop_size=crop_size, interpolation=interpolation
    )

    score = BENCHMARKS[benchmark_identifier](brain_model)
    score = postprocess_score(score)

    return score


def main(args):
    reproducibility.seed_all(args.seed)
    
    print(args.run_name)
    if args.run_name.replace("_imagenet_full", "") in ABLATION:
        model_identifier = args.run_name.replace("_imagenet_full", "")
    elif args.use_open_clip:
        model_identifier = args.run_name
    elif args.use_timm:
        model_identifier = args.arch
    elif args.model_commitment:
        model_identifier =  args.model_commitment.split(':')[0]
    else:
        model_identifier = MODEL_IDENTIFIERS.get(args.arch, args.arch)
        
    if args.commitment_file:
        with open(args.commitment_file, 'r') as f:
            MODEL_COMMITMENTS_ = json.load(f)
    else:
        MODEL_COMMITMENTS_ = MODEL_COMMITMENTS
    
    print(f"Model identifier: {model_identifier}")
    if 'cornet_s' in model_identifier.lower():
        pass
    elif model_identifier not in MODEL_COMMITMENTS_:
        raise ValueError(f"Model {model_identifier} not found in 'commitments.json'")
    
    model = create_model(**vars(args))
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

    training_dataset = f"{args.training_dataset}_{args.ckpt_src}"
    
    pattern = re.compile(r'seed-\d+')
    match = pattern.search(args.run_name)
    if match:
        seed = int(match.group().split('-')[1])
    else:
        seed = 0
    
    if args.use_timm:
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
    elif args.use_open_clip:
        transforms = model.preprocess
        # OpenCLIP models permute the axes, so we need to undo that
        postprocess_func = lambda x: x.permute(1, 0, 2)  # LND -> NLD
        layers = MODEL_COMMITMENTS_[model_identifier][training_dataset]['seed-0']['layers']
        layers.append(MODEL_COMMITMENTS_[model_identifier][training_dataset]['seed-0']['behavioral_readout_layer'])
        print(layers)
        model = PostprocessWrapper(model, layers, postprocess_func, skip_layers=[layers[-1]], layer_names_modified=True)
        layers = model.layers_
    else:
        transforms = None
    
    if args.model_commitment:
        _model_identifier, _training_dataset, _seed = args.model_commitment.split(':')
        model_commitments = MODEL_COMMITMENTS_[_model_identifier][_training_dataset]
        if f"seed-{seed}" not in model_commitments:
            print(f"Model commitment for seed-{_seed} not found, using seed-0")
            _seed = 0
        model_commitment = model_commitments[f"seed-{seed}"]
    else:
        if 'cornet_s' in model_identifier.lower():
            model_commitment = None
        else:
            model_commitments = MODEL_COMMITMENTS_[model_identifier][training_dataset]
            if f"seed-{seed}" not in model_commitments:
                print(f"Model commitment for seed-{seed} not found, using seed-0")
                seed = 0
            model_commitment = model_commitments[f"seed-{seed}"]
            
            [f"seed-{seed}"]
    print('Model commitment', model_commitment)

    for region, benchmark_identifier in tqdm(BENCHMARKS_IDENTIFIERS.items(), desc="Processing benchmarks"):
        score = score_model(
            model_identifier, 
            model, 
            model_commitment, 
            region,
            transforms=transforms,
            resize_size=args.resize_size, 
            crop_size=args.crop_size, 
            interpolation=args.interpolation
            )
        print(score)
        scores[benchmark_identifier] = list(score.values())
        print(f"{benchmark_identifier}: {scores[benchmark_identifier]}")

    if args.suffix:
        suffix = f"_{args.suffix}"
    else:
        suffix = ""
    
    logs = {}
    logs["scores"] = scores   
    logs["args"] = vars(args)
    logs["model_commitment"] = model_commitment
        
    save_path = Path(args.save_dir) / f"{args.run_name}{suffix}.json"
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=False)
    with open(save_path, "w") as f:
        json.dump(logs, f, indent=4)
        
    print(scores)
    print(f"Saved scores to {save_path}")


if __name__ == "__main__":
    parser = create_argparser()
    
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
        
    parser.set_defaults(save_dir='./outputs/benchmark_results')
    args = parser.parse_args(remaining)
    print(args)
    main(args)

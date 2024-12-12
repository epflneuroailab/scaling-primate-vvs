from pathlib import Path
from typing import List, Literal
import logging
import json
import yaml

import pandas as pd

from composer import ComposerModel
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from scaling_primate_vvs.brainscore.src.utils import wrap_brain_model

artifact_dir = Path(__file__).parent

SUPPORTED_MODELS = []
SUPPORTED_MODELS += [f'resnet{k}' for k in [18, 34, 50, 101, 152]]
SUPPORTED_MODELS += [f'efficient_b{k}' for k in [0,1,2]]
SUPPORTED_MODELS += [f'convnext_{k}' for k in ['tiny', 'small', 'medium', 'large']]
SUPPORTED_MODELS += [f'deit_{k}' for k in ['tiny', 'small', 'medium', 'large']]
SUPPORTED_MODELS += ['alexnet', 'cornet_s']

training_configs_dir = Path(__file__).parent.parent / 'training/cfgs/runai'
MODEL_CONFIGS = {}
MODEL_CONFIGS.update({
    f'resnet{k}': training_configs_dir / 'resnet' / f'resnet{k}.yaml' for k in [18, 34, 50, 101, 152]
})
MODEL_CONFIGS.update({
    f'efficient_b{k}': training_configs_dir / 'efficientnet' / f'efficientnet_b{k}.yaml' for k in [0,1,2]
})
MODEL_CONFIGS.update({
    f'convnext_{k}': training_configs_dir / 'convnext' / f'convnext_{k}.yaml' for k in ['tiny', 'small', 'medium', 'large']
})
MODEL_CONFIGS.update({
    f'deit_{k}': training_configs_dir / 'deit' / f'deit_{k}.yaml' for k in ['tiny', 'small', 'medium', 'large']
})
MODEL_CONFIGS.update({
    'alexnet': training_configs_dir / 'others' / 'alexnet.yaml'
})
MODEL_CONFIGS.update({
    'cornet_s': training_configs_dir / 'others' / 'cornet_s.yaml'
})
NAME_MAPPING = {f'deit_{k}': f'deit3_{k}_patch16_224' for k in ['tiny', 'small', 'medium', 'large']}


try:
    DF_RESULTS = pd.read_csv(artifact_dir / 'benchmark_scores.csv')
except FileNotFoundError:
    DF_RESULTS = None
    logging.warning("No `results.csv` file found in the `scaling_primate_vvs/artifacts` directory.")
    
try:
    DF_MODELS = pd.read_csv(artifact_dir / 'model_checkpoints.csv')
except FileNotFoundError:
    DF_MODELS = None
    logging.warning("No `model_checkpoints.csv` file found in the `scaling_primate_vvs/artifacts` directory.")

def list_models() -> List[str]:
    if DF_MODELS is None:
        print("`model_checkpoints.csv` file not found.")
        return []
    return DF_MODELS['model_id'].tolist()

def get_model_checkpoints_dataframe() -> pd.DataFrame:
    if DF_MODELS is None:
        print("`model_checkpoints.csv` file not found.")
    
    return DF_MODELS

def get_benchmark_scores_dataframe(benchmark_type:Literal['public', 'private']) -> pd.DataFrame:
    if benchmark_type == 'public':
        if DF_RESULTS is None:
            print("`benchmark_scores.csv` file not found.")
        df = DF_RESULTS.copy()
        
    elif benchmark_type == 'private':
        try:
            df = pd.read_csv(artifact_dir / 'benchmark_scores_brainscore.csv')
        except FileNotFoundError:
            print("`benchmark_scores_brainscore.csv` file not found.")
            df = None

    return df


def deep_update(d, u):  
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_yaml(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
        
    base_config = config.get('defaults', None)
    if base_config is not None:
        configs_dir = Path(__file__).parent.parent / 'training'
        base_config = configs_dir / base_config
        defaults = load_yaml(base_config)
        config = deep_update(defaults, config)

    return config

def load_model(model_id:str, checkpoints_dir:str = None) -> ComposerModel:
    if model_id not in list_models():
        raise ValueError(f"Model with id {model_id} not found")
    
    assert DF_RESULTS is not None, "No `benchmark_scores.csv` file found."
    
    model_info = DF_MODELS[DF_MODELS['model_id'] == model_id].iloc[0]
    arch = model_info['arch']
    if arch not in SUPPORTED_MODELS:
        raise ValueError(f"Model with arch {arch} not supported by this utility loader.")

    from scaling_primate_vvs.training import create_model
    
    load_model_ema = False
    if ('convnext' in arch) or ('deit' in arch):
        load_model_ema = True
        
    try:
        training_config_path = MODEL_CONFIGS[arch]
        training_config = load_yaml(training_config_path)
    except FileNotFoundError:
        raise ValueError(f"Training config for model with arch {arch} cannot be loaded.")
    
    model_kwargs = training_config
    assert model_kwargs['arch'] == NAME_MAPPING.get(arch, arch), f"Model arch mismatch: {model_kwargs['arch']} != {arch}"
       
    
    model_kwargs['load_model_ema'] = load_model_ema
    
    
    if checkpoints_dir is not None:
        checkpoint_info = DF_MODELS[DF_MODELS['model_id'] == model_id].iloc[0]
        checkpoint = Path(checkpoints_dir) / checkpoint_info['checkpoint_path']
    else:
        checkpoint = model_info['checkpoint_url']
        
    model_kwargs['checkpoint'] = checkpoint
    
    model = create_model(**model_kwargs)
    
    return model


def load_brain_model(model_id:str, checkpoints_dir:str = None) -> ModelCommitment:
    model = load_model(model_id, checkpoints_dir)
    
    brainsscore_artifacts_dir = Path(__file__).parent.parent / 'brainscore/artifacts'
    model_commitment_file = brainsscore_artifacts_dir / 'commitments.json'
    try:
        with open(model_commitment_file, 'r') as f:
            model_commitments = json.load(f)
            model_commitments = {k.lower(): v for k,v in model_commitments.items()}
    except FileNotFoundError:
        logging.warning(f"File {model_commitment_file} not found.")
        model_commitments = {}
        
    model_info = DF_RESULTS[DF_RESULTS['model_id'] == model_id].iloc[0]
    arch = model_info['arch']
    if arch in model_commitments:
        dataset = model_info['dataset']
        
        model_commitment = model_commitments[arch][f'{dataset}_full']['seed-0']    
    
    else:
        logging.warning(f"Model commitment for model with id {model_id} not found.")
        model_commitment = {}
        
    try:
        training_config_path = MODEL_CONFIGS[arch]
        training_config = load_yaml(training_config_path)
    except FileNotFoundError:
        raise ValueError(f"Training config for model with arch {arch} cannot be loaded.")

    brain_model = wrap_brain_model(
        identifier=model_id,
        model=model,
        resize_size=training_config['val_resize_size'],
        crop_size=training_config['val_crop_size'],
        interpolation=training_config.get('interpolation', 'bilinear'),
        model_commitment=model_commitment
    )
    

    return brain_model
        
    
    
    
    
    
    
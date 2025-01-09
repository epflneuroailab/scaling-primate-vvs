from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
import logging
import json
import yaml

import pandas as pd

from composer import ComposerModel
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from scaling_primate_vvs.brainscore.src.utils import wrap_brain_model

# Define the directory where artifacts are stored, based on the current file's location
artifact_dir = Path(__file__).parent

# List of supported model architectures
SUPPORTED_MODELS: List[str] = []
SUPPORTED_MODELS += [f'resnet{k}' for k in [18, 34, 50, 101, 152]]
SUPPORTED_MODELS += [f'efficientnet_b{k}' for k in [0, 1, 2]]
SUPPORTED_MODELS += [f'convnext_{k}' for k in ['tiny', 'small', 'base', 'large']]
SUPPORTED_MODELS += [f'deit_{k}' for k in ['tiny', 'small', 'base', 'large']]
SUPPORTED_MODELS += ['alexnet', 'cornet_s']

# Directory containing training configuration YAML files
training_configs_dir = Path(__file__).parent.parent / 'training/cfgs/runai'

# Mapping from model names to their corresponding configuration file paths
MODEL_CONFIGS: Dict[str, Path] = {}
MODEL_CONFIGS.update({
    f'resnet{k}': training_configs_dir / 'resnet' / f'resnet{k}.yaml' for k in [18, 34, 50, 101, 152]
})
MODEL_CONFIGS.update({
    f'efficientnet_b{k}': training_configs_dir / 'efficientnet' / f'efficientnet_b{k}.yaml' for k in [0, 1, 2]
})
MODEL_CONFIGS.update({
    f'convnext_{k}': training_configs_dir / 'convnext' / f'convnext_{k}.yaml' for k in ['tiny', 'small', 'base', 'large']
})
MODEL_CONFIGS.update({
    f'deit_{k}': training_configs_dir / 'deit' / f'deit_{k}.yaml' for k in ['tiny', 'small', 'base', 'large']
})
MODEL_CONFIGS.update({
    'alexnet': training_configs_dir / 'others' / 'alexnet.yaml',
    'cornet_s': training_configs_dir / 'others' / 'cornet_s.yaml'
})

# Mapping from original model names to their corresponding names in the configuration
NAME_MAPPING: Dict[str, str] = {
    f'deit_{k}': f'deit3_{k}_patch16_224' for k in ['tiny', 'small', 'base', 'large']
}

DATASET_CLASSES = {
    'imagenet': 1000,
    'ecoset': 565,
    'imagenet21kP': 10450,
    'webvisionP': 4186,
    'places365': 365,
    'inaturalist': 10000,
    'infimnist': 10
}

# Attempt to load benchmark results from CSV; log a warning if the file is not found
try:
    DF_RESULTS: Optional[pd.DataFrame] = pd.read_csv(artifact_dir / 'benchmark_scores.csv')
except FileNotFoundError:
    DF_RESULTS = None
    logging.warning("No `benchmark_scores.csv` file found in the `scaling_primate_vvs/artifacts` directory.")

# Attempt to load model checkpoints from CSV; log a warning if the file is not found
try:
    DF_MODELS: Optional[pd.DataFrame] = pd.read_csv(artifact_dir / 'model_checkpoints.csv')
except FileNotFoundError:
    DF_MODELS = None
    logging.warning("No `model_checkpoints.csv` file found in the `scaling_primate_vvs/artifacts` directory.")


def list_models(supported_simple_loading:bool = True) -> List[str]:
    """
    Retrieve a list of available model IDs from the model checkpoints dataframe.
    
    Args:
        supported_simple_loading (True): A boolean flag to determine whether to use the supported models list for
         loading via this utility. If set to False, the model checkpoints availabe on aws s3 bucket will be returned.

    Returns:
        List[str]: A list of model IDs. Returns an empty list if the model checkpoints file is not found.
    """
    if DF_MODELS is None:
        print("`model_checkpoints.csv` file not found.")
        return []
    if supported_simple_loading:
        df = DF_MODELS[DF_MODELS['arch'].isin(SUPPORTED_MODELS)]
        return df['model_id'].unique().tolist()
    
    else:
        return DF_MODELS['model_id'].unique().tolist()


def get_model_checkpoints_dataframe() -> Optional[pd.DataFrame]:
    """
    Get the dataframe containing model checkpoints information.

    Returns:
        Optional[pd.DataFrame]: The dataframe with model checkpoints, or None if the file is not found.
    """
    if DF_MODELS is None:
        print("`model_checkpoints.csv` file not found.")
        return None
    return DF_MODELS


def get_benchmark_scores_dataframe(benchmark_type: Literal['public', 'private']) -> Optional[pd.DataFrame]:
    """
    Retrieve the benchmark scores dataframe based on the specified benchmark type.

    Args:
        benchmark_type (Literal['public', 'private']): The type of benchmark scores to retrieve.

    Returns:
        Optional[pd.DataFrame]: The dataframe containing benchmark scores, or None if the file is not found.
    """
    if benchmark_type == 'public':
        if DF_RESULTS is None:
            print("`benchmark_scores.csv` file not found.")
            return None
        return DF_RESULTS.copy()
    elif benchmark_type == 'private':
        try:
            df = pd.read_csv(artifact_dir / 'benchmark_scores_brainscore.csv')
            return df
        except FileNotFoundError:
            print("`benchmark_scores_brainscore.csv` file not found.")
            return None
    else:
        print(f"Unsupported benchmark type: {benchmark_type}")
        return None


def deep_update(d: Dict[Any, Any], u: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively update dictionary `d` with values from dictionary `u`.

    Args:
        d (Dict[Any, Any]): The original dictionary to be updated.
        u (Dict[Any, Any]): The dictionary containing updates.

    Returns:
        Dict[Any, Any]: The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_yaml(yaml_file: str) -> Dict[Any, Any]:
    """
    Load a YAML configuration file and recursively apply default configurations if specified.

    Args:
        yaml_file (str): The path to the YAML file to load.

    Returns:
        Dict[Any, Any]: The loaded and merged configuration dictionary.
    """
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Check for default configurations to merge
    base_config = config.get('defaults', None)
    if base_config is not None:
        # Define the base configurations directory
        configs_dir = Path(__file__).parent.parent / 'training'
        base_config_path = configs_dir / base_config

        # Recursively load the base configuration
        defaults = load_yaml(str(base_config_path))
        config = deep_update(defaults, config)

    return config


def load_model(model_id: str, checkpoints_dir: Optional[str] = None) -> ComposerModel:
    """
    Load a ComposerModel based on the provided model ID and optional checkpoints directory.

    Args:
        model_id (str): The identifier of the model to load.
        checkpoints_dir (Optional[str], optional): The directory containing model checkpoints. Defaults to None.

    Raises:
        ValueError: If the model ID is not found or if the model architecture is unsupported.
        AssertionError: If the benchmark scores dataframe is not loaded.

    Returns:
        ComposerModel: The loaded ComposerModel instance.
    """
    # Verify that the model ID exists
    available_models = list_models()
    if model_id not in available_models:
        raise ValueError(f"Model with id {model_id} not found")

    # Ensure that benchmark scores are available
    assert DF_RESULTS is not None, "No `benchmark_scores.csv` file found."

    # Retrieve model information from the dataframe
    model_info = DF_RESULTS[DF_RESULTS['model_id'] == model_id].iloc[0]
    arch = model_info['arch']

    # Check if the architecture is supported
    if arch not in SUPPORTED_MODELS:
        raise ValueError(f"Model with arch {arch} not supported by this utility loader.")

    from scaling_primate_vvs.training import create_model

    # Determine whether to load the Exponential Moving Average (EMA) of the model weights
    load_model_ema = False
    if ('convnext' in arch) or ('deit' in arch):
        load_model_ema = True

    # Load the training configuration for the model
    try:
        training_config_path = str(MODEL_CONFIGS[arch])
        training_config = load_yaml(training_config_path)
    except FileNotFoundError:
        raise ValueError(f"Training config for model with arch {arch} cannot be loaded.")

    # Ensure that the architecture in the config matches the expected architecture
    expected_arch = NAME_MAPPING.get(arch, arch)
    if training_config.get('arch') != expected_arch:
        raise ValueError(f"Model arch mismatch: {training_config.get('arch')} != {arch}")

    # Update the configuration with EMA loading preference
    training_config['load_model_ema'] = load_model_ema

    # Determine the checkpoint path
    if checkpoints_dir is not None:
        checkpoint_info = DF_MODELS[DF_MODELS['model_id'] == model_id]
        checkpoint_info = checkpoint_info.sort_values('epoch', ascending=False).iloc[0]
        checkpoint = Path(checkpoints_dir) / checkpoint_info['checkpoint_path']
        checkpoint = str(checkpoint)
    else:
        checkpoint = model_info['checkpoint_url']

    # Add the checkpoint path to the configuration
    training_config['checkpoint'] = checkpoint

    # Add the number of classes for the dataset to the configuration
    dataset = model_info['dataset']
    num_classes = DATASET_CLASSES.get(dataset, 1000)
    training_config['num_classes'] = num_classes

    # Create and return the ComposerModel instance
    model = create_model(**training_config)

    return model


def load_brain_model(model_id: str, checkpoints_dir: Optional[str] = None) -> ModelCommitment:
    """
    Load a brain model wrapped with brain region commitments based on the provided model ID.

    Args:
        model_id (str): The identifier of the model to load.
        checkpoints_dir (Optional[str], optional): The directory containing model checkpoints. Defaults to None.

    Raises:
        ValueError: If the training configuration for the model architecture cannot be loaded.

    Returns:
        ModelCommitment: The wrapped brain model with commitments.
    """
    # Load the base model using the load_model function
    model = load_model(model_id, checkpoints_dir)
    
    # Unwrap the model to access the underlying model
    if hasattr(model, "module"):
        model = model.module
    elif hasattr(model, "model"):
        model = model.model
    else:
        raise ValueError("Model cannot be unwrapped.")

    # Define the directory containing brainscore artifacts
    brainscore_artifacts_dir = Path(__file__).parent.parent / 'brainscore/artifacts'
    model_commitment_file = brainscore_artifacts_dir / 'commitments.json'

    # Attempt to load model commitments from the JSON file
    try:
        with open(model_commitment_file, 'r') as f:
            model_commitments = json.load(f)
            # Normalize keys to lowercase for consistent access
            model_commitments = {k.lower(): v for k, v in model_commitments.items()}
    except FileNotFoundError:
        logging.warning(f"File {model_commitment_file} not found.")
        model_commitments = {}

    # Retrieve model information from the results dataframe
    model_info = DF_RESULTS[DF_RESULTS['model_id'] == model_id].iloc[0]
    arch = model_info['arch']

    if 'cornet' in arch:
        model_commitment = {
            'region2layer': {
                region: f'{region}.output' for region in ['V1', 'V2', 'V4', 'IT']
            },
            'layers': [
                f'{region}.output' for region in ['V1', 'V2', 'V4', 'IT']
            ] + ['decoder.avgpool'],
            'behavioral_readout_layer': 'decoder.avgpool'
        }

    # Check if the architecture has a corresponding commitment
    elif arch in model_commitments:
        dataset = model_info['dataset']
        # Retrieve the specific commitment for the dataset and seed
        model_commitment = model_commitments[arch][f'{dataset}_full']['seed-0']
    else:
        logging.warning(f"Model commitment for model with id {model_id} not found.")
        model_commitment = {}

    # Load the training configuration for the model
    try:
        training_config_path = str(MODEL_CONFIGS[arch])
        training_config = load_yaml(training_config_path)
    except FileNotFoundError:
        raise ValueError(f"Training config for model with arch {arch} cannot be loaded.")

    # Wrap the loaded model with brain-specific transformations
    brain_model = wrap_brain_model(
        identifier=model_id,
        model=model,
        resize_size=training_config.get('val_resize_size', 224),  # Default resize size if not specified
        crop_size=training_config.get('val_crop_size', 224),      # Default crop size if not specified
        interpolation=training_config.get('interpolation', 'bilinear'),
        model_commitment=model_commitment
    )

    return brain_model

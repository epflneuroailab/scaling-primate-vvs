import argparse
import yaml
import hashlib
from pathlib import Path

DATASET_NAMES = {
    'imagenet': 'ImageNet-1k',
    'ecoset': 'ecoset',
    'imagenet21kP': 'ImageNet21kP',
    'imagenet21kP-class-1000': 'ImageNet21kP-Class 1000',
    'webvision': 'WebVision',
    'webvisionP': 'WebVisionP',
    'laion': 'LAION',
}

BENCHMARKS = [
        'movshon.FreemanZiemba2013public.V1-pls',
        'movshon.FreemanZiemba2013public.V2-pls',
        'dicarlo.MajajHong2015public.V4-pls',
        'dicarlo.MajajHong2015public.IT-pls',
        'dicarlo.Rajalingham2018public-i2n'
    ]

REGION2BENCHMARKS = {
    'V1':'movshon.FreemanZiemba2013public.V1-pls',
    'V2':'movshon.FreemanZiemba2013public.V2-pls',
    'V4':'dicarlo.MajajHong2015public.V4-pls',
    'IT':'dicarlo.MajajHong2015public.IT-pls',
    'Behavioral': 'dicarlo.Rajalingham2018public-i2n'
}


def deep_update(d, u):
    """
    Recursively updates a dictionary `d` with another dictionary `u`.
    If the value of a key in `u` is a dictionary, the function will recursively update
    the corresponding dictionary in `d`. Otherwise, it will directly update the value
    in `d` with the value from `u`.
    Parameters:
    d (dict): The dictionary to be updated.
    u (dict): The dictionary with updates.
    Returns:
    dict: The updated dictionary `d`.
    """
    
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_yaml(yaml_file: str) -> dict:
    """
    Loads a yaml file and returns the content as a dictionary.
    
    Parameters:
    yaml_file (str): The path to the yaml file to be loaded.
    
    Returns:
    dict: The content of the yaml file as a dictionary.
    """
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
        
    base_config = config.get('base_config', None)
    if base_config is not None:
        yaml_file = Path(yaml_file)
        base_config = yaml_file.parent / base_config
        # Load a base configuration files if exists
        defaults = load_yaml(base_config)
        # Update the defaults with the current configuration
        config = deep_update(defaults, config)

    return config

def get_md5_hash(data: dict) -> str:
    """
    Calculates the MD5 hash of the input data.
    """
    data_str = str(data)
    return hashlib.md5(data_str.encode()).hexdigest()



def get_args():
    argparser = argparse.ArgumentParser(description="Bootstrapping for scaling law curve fittings")
    
    argparser.add_argument(
        "--experiment-config",
        dest="experiment_config",
        type=str,
        default=None,
        help="Path to the yaml file containing the experiment configuration."
    )
    
    argparser.add_argument(
        "--results-csv",
        dest="results_csv",
        type=str,
        default="../../../artifacts/benchmark_scores_local.csv",
        help="Path to the csv file containing the experimental results."
    )
    
    argparser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="./fitting_results",
        help="Path to the output directory."
    )
    argparser.add_argument(
        "--artifact-dir",
        dest="artifact_dir",
        type=str,
        default="./fitting_results",
        help="Path to bootstrapped results directory."
    )
    argparser.add_argument(
        "--experiment-name",
        dest="experiment_name",
        type=str,
        default=None,
        help="Name of the experiment."
    )
    argparser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=8,
        help="Number of workers to use for multiprocessing (default is 8)."
    )
    argparser.add_argument(
        "--overwrite",
        dest="overwrite",
        action='store_true',
        help="Whether to overwrite existing files."
    )
    argparser.add_argument(
        "--num-bootstraps",
        dest="num_bootstraps",
        type=int,
        default=None,
        help="Number of bootstraps to perform. The default is None, which is the intended use for using the value from the configuration file." \
                "If a value is provided, it will override the value from the configuration file." \
                "Useful for testing purposes."
    )
    
    return argparser



# Scaling Laws for Task-Optimized Models of the Primate Visual Ventral Stream

[![arXiv](https://img.shields.io/badge/arXiv-2411.05712-b31b1b.svg)](https://arxiv.org/abs/2411.05712)

This repository contains code, links to model checkpoints and benchmark results for our study on scaling laws in computational models of the primate visual system. We systematically evaluate over 600 models to understand how model size, dataset size, and compute resources impact alignment with brain function and behavior.

## Overview

Our study explores:
- Scaling laws for brain and behavioral alignment
- Impact of model architecture and dataset quality
- Optimal compute allocation between model and data scaling
- Effects of scale across different brain regions

## Repository Structure

```
scaling_primate_vvs/
      ├── analysis/            # Scripts for curve fitting and statistical analysis
      ├── brainscore/          # Brain-Score benchmarks and layer selection for region commitments
      ├── artifacts/             # Benchmark results and info on model checkpoints
      ├── training/            # Model training scripts and configurations
      └── visualization/       # Plotting scripts and notebooks
```

Model weight can be downloaded as follows:

```bash
S3_URL="https://epfl-neuroailab-scalinglaws.s3.eu-north-1.amazonaws.com/checkpoints"
MODEL_ID="resnet18_imagenet_full"
CHECKPOINT=100

wget ${S3_URL}/${MODEL_ID}/ep${CHECKPOINT}.pt
```


The package contains several utility functions that can be used to load models, retrieve benchmark scores, and access model checkpoint information. Here's a brief overview of the main functions:

- `list_models()`: Returns a list of available model IDs.
- `get_model_checkpoints_dataframe()`: Returns a DataFrame containing model checkpoint information.
- `get_benchmark_scores_dataframe(benchmark_type)`: Returns a DataFrame containing benchmark scores. The `benchmark_type` parameter can be either `'public'` or `'private'`.
- `load_model(model_id)`: Loads a model given its ID.
- `load_brain_model(model_id)`: Loads a brain model with region-wise commitments.

### Example Usage

```python
from scaling_primate_vvs import list_models, load_brain_model, load_model, get_benchmark_scores_dataframe, get_model_checkpoints_dataframe

# List all available models
available_models = list_models()
print("Available Models:", available_models)

# Load a specific model
model_id = 'resnet18_imagenet_full'
model = load_model(model_id)


# Load a brain-mapped model
brain_model = load_brain_model(model_id)

# Retrieve benchmark scores on public benchmarks
public_benchmarks = get_benchmark_scores_dataframe('public')
print(public_benchmarks[public_benchmarks['model_id'] == model_id])
```

## Installation Guide

Follow these steps to install the `scaling_primate_vvs` package:

1. Ensure that you have Python 3.11 or later installed on your system.

2. Install the package using one of the following methods:

      2.1. Install directly from the GitHub repository:
      ```bash
      pip install git+https://github.com/epflneuroailab/scaling-primate-vvs.git
      ```

      2.2. Clone the repository:

      ```bash
      git clone https://github.com/epflneuroailab/scaling-primate-vvs.git
      cd scaling_primate_vvs
      ```

      Install the package using one of the following methods:

      a. Basic installation:

      ```bash
      pip install .
      ```

      b. Installation with `brainscore_vision` dependencies:

      ```bash
      pip install ".[brainscore]"
      ```

      c. Installation with training dependencies:

      ```bash
      pip install ".[training]"
      ```

      d. Installation with all dependencies:

      ```bash
      pip install ".[all]"
      ```

3. Verify the installation by running the following command:
   ```
   python -c "import scaling_primate_vvs"
   ```

   If no errors are raised, the installation was successful.

### Pitfalls and Considerations

- The `brainscore_vision` package is installed from a forked GitHub repository. This is due to compatibility issues where `brainscore_vision` requires `importlib_metadata<5` while `composer` requires `importlib_metadata>=5`. Installing this forked version may lead to unexpected behavior, so proceed with caution.

- The `training` dependencies include `lightly` and `torchattacks`, which are required for training models in self-supervised and adversarial settings. Make sure to install these dependencies if you intend to train models.


### Citation
```
@article{gokce2024scalingprimatevvs,
      title={Scaling Laws for Task-Optimized Models of the Primate Visual Ventral Stream}, 
      author={Abdulkadir Gokce and Martin Schrimpf},
      year={2024},
      eprint={2411.05712},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2411.05712}, 
}
```

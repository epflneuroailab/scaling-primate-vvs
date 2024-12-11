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
├── analysis/            # Scripts for curve fitting and statistical analysis
├── brainscore/          # Brain-Score benchmarks and layer selection for region commitments
├── results/             # Benchmark results
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
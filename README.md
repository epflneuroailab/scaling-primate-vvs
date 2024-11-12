# Scaling Laws for Task-Optimized Models of the Primate Visual Ventral Stream

[![arXiv](https://img.shields.io/badge/arXiv-2406.15109-b31b1b.svg)](https://arxiv.org/abs/2411.05712v1)

### Abstract
>  When trained on large-scale object classification datasets, certain artificial neural network models begin to approximate core object recognition (COR) behaviors and neural response patterns in the primate visual ventral stream (VVS). While recent machine learning advances suggest that scaling model size, dataset size, and compute resources improve task performance, the impact of scaling on brain alignment remains unclear. In this study, we explore scaling laws for modeling the primate VVS by systematically evaluating over 600 models trained under controlled conditions on benchmarks spanning V1, V2, V4, IT and COR behaviors. We observe that while behavioral alignment continues to scale with larger models, neural alignment saturates. This observation remains true across model architectures and training datasets, even though models with stronger inductive bias and datasets with higher-quality images are more compute-efficient. Increased scaling is especially beneficial for higher-level visual areas, where small models trained on few samples exhibit only poor alignment. Finally, we develop a scaling recipe, indicating that a greater proportion of compute should be allocated to data samples over model size. Our results suggest that while scaling alone might suffice for alignment with human core object recognition behavior, it will not yield improved models of the brain's visual ventral stream with current architectures and datasets, highlighting the need for novel strategies in building brain-like models.

### Status
The repository is currently under construction. You can find the benchmark results under the `results` folder. Model weight can be downloaded as follows:

```bash
S3_URL="https://epfl-neuroailab-scalinglaws.s3.eu-north-1.amazonaws.com/checkpoints"
MODEL_ID="resnet18_imagenet_full"
CHECKPOINT=100

wget ${S3_URL}/${MODEL_ID}/ep${CHECKPOINT}.pt
```


#### Progress
- [x] Benchmark results
- [x] Model weights
- [ ] Analysis
- [ ] Visualization
- [ ] Model training
- [ ] Benchmarking scripts


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
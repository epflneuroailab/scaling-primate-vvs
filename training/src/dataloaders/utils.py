from collections import defaultdict

import numpy as np

from torch.utils.data import DistributedSampler

from timm.data.distributed_sampler import RepeatAugSampler

def subsample_dataset(dataset, num_samples, sample_strategy="per_class", class_id_map=None):
    """Subsample a dataset.

    Args:
        dataset (Dataset): The dataset to be subsampled.
        num_samples (int): The number of samples to be selected.
        sample_strategy (str, optional): The strategy for subsampling. 
            Defaults to "per_class".
        class_id_map (dict, optional): A dictionary mapping old class ids to new class ids.

    Returns:
        Dataset: The subsampled dataset.

    Raises:
        ValueError: If an unknown sample strategy is provided.

    Note:
        This function modifies the dataset in-place.
        The subsampling can be done either per class or for the whole dataset.
        If `sample_strategy` is set to "per_class", each class will have the same number of samples.
        If `sample_strategy` is set to "whole_dataset", the samples will be randomly selected from the entire dataset.
        If `sample_strategy` is set to "sample_classes", a subset of classes will be randomly selected.

    """

    target_indices = defaultdict(list)
    for index, target in enumerate(dataset.targets):
        target_indices[target].append(index)

    sampled_indices = []
    if sample_strategy == "per_class":
        for target, indices in target_indices.items():
            assert (
                len(indices) >= num_samples
            ), f"Class {target} has only {len(indices)} samples, but {num_samples} were requested."
            indices = np.random.choice(indices, size=num_samples, replace=False)
            sampled_indices.extend(indices)

        dataset.samples = [dataset.samples[i] for i in sampled_indices]
        dataset.targets = [dataset.targets[i] for i in sampled_indices]
    elif sample_strategy == "whole_dataset":
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        dataset.samples = [dataset.samples[i] for i in indices]
        dataset.targets = [dataset.targets[i] for i in indices]
    elif sample_strategy == "sample_classes":
        if class_id_map is not None:
            dataset.samples = [
                (path, class_id_map[target]) for path, target in dataset.samples
                if target in class_id_map
            ]
            dataset.targets = [class_id_map[target] for target in dataset.targets if target in class_id_map]
            dataset.new_class_ids = class_id_map
            dataset.old_class_ids = {new: old for old, new in class_id_map.items()}
        else:
            assert num_samples <= len(target_indices), \
                f"Only {len(target_indices)} classes available, but {num_samples} were requested." 
            # Randomly select a subset of classes
            sampled_classes = np.random.choice(
                list(target_indices.keys()), size=num_samples, replace=False
            )
            # Map old class ids to new class ids
            new_class_ids = {old: new for new, old in enumerate(sampled_classes)}
            # Filter out samples from classes that were not selected, and remap class ids
            dataset.samples = [
                (path, new_class_ids[target]) for path, target in dataset.samples
                if target in sampled_classes
            ]
            # Remap class ids
            dataset.targets = [new_class_ids[target] for target in dataset.targets if target in sampled_classes]
            dataset.new_class_ids = new_class_ids
            dataset.old_class_ids = {new: old for old, new in new_class_ids.items()}
    
    else:
        raise ValueError(f"Unknown sample strategy {sample_strategy}")

    return dataset

class RepeatAugSamplerV2(DistributedSampler, RepeatAugSampler):
    def __init__(self, *args, **kwargs):
        DistributedSampler.__init__(self, *args, **kwargs)
        RepeatAugSampler.__init__(self, *args, **kwargs)
        
    def __iter__(self):
        return RepeatAugSampler.__iter__(self)
    
    def __len__(self):
        return RepeatAugSampler.__len__(self)
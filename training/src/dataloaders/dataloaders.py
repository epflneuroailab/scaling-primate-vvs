from typing import Union, List, Tuple
from pathlib import Path
from collections import defaultdict
import warnings
import math

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision

from composer.utils import dist


from .transforms import create_transforms
from .datasets import ImageFolder_Albumentations, DatasetFileList, DatasetWrapper
from .utils import subsample_dataset, RepeatAugSamplerV2


def create_dataloaders(**kwargs):
    root = kwargs.get("data_path")
    dataset = kwargs.get("dataset")
    batch_size = kwargs.get("batch_size")
    num_workers = kwargs.get("workers")
    pin_memory = kwargs.get("pin_memory")
    train_samples = kwargs.get("train_samples")
    sample_strategy = kwargs.get("sample_strategy")
    subsample_classes = kwargs.get("subsample_classes")
    drop_last_train = kwargs.get("drop_last_train")
    transform_lib = kwargs.get("transform_lib")
    pytorch_aug_set = kwargs.get("pytorch_aug_set")
    repeated_aug = kwargs.get("repeated_aug")
    
    root = root if isinstance(root, Path) else Path(root)

    transform_train, transform_val = create_transforms(**kwargs)
    if transform_lib == "albumentations":
        imageFolder = ImageFolder_Albumentations
    else:
        imageFolder = ImageFolder
    # If kNN task, do not drop the last batch to ensure that all samples are used for training
    if drop_last_train is None:
        drop_last_train = False if (transform_lib and pytorch_aug_set == "kNN") else True

    if dataset in ["imagenet", "ecoset"]:
        ds_train = imageFolder(root / "train", transform=transform_train)
        ds_eval = imageFolder(root / "val", transform=transform_val)
    elif dataset in ["webvision", "imagenet21k", "imagenet21kP", "webvisionP"]:
        ds_train = DatasetFileList(root / "images" / "train", root /  "metadata" / "train_file_list.txt", transform=transform_train)
        ds_eval = DatasetFileList(root / "images" / "val", root / "metadata" / "val_file_list.txt", transform=transform_val)
    elif dataset == "mnist":
        transform = T.Compose([
            T.Lambda(lambda x: x.convert('RGB') ),
            # T.HFlip(),
            T.ToTensor(),
            # T.transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds_train = torchvision.datasets.MNIST(root, train=True, transform=transform, download=True)
        ds_eval = torchvision.datasets.MNIST(root, train=False, transform=transform, download=True)
    elif dataset == "cifar10":
        transform_train = T.Compose([
            # T.HFlip(),
            T.ToTensor(),
            # T.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        transform_train = T.Compose([
            # T.HFlip(),
            T.ToTensor(),
            # T.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        ds_train = torchvision.datasets.CIFAR10(root, train=True, transform=transform_train, download=True)
        ds_eval = torchvision.datasets.CIFAR10(root, train=False, transform=transform_val, download=True)
    elif dataset == "cifar100":
        transform = T.Compose([
            # T.HFlip(),
            T.ToTensor(),
            # T.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.267, 0.256, 0.276))
        ])
        ds_train = torchvision.datasets.CIFAR100(root, train=True, transform=transform, download=True)
        ds_eval = torchvision.datasets.CIFAR100(root, train=False, transform=transform, download=True)
    elif dataset == "svhn":
        transform = T.Compose([
            # T.HFlip(),
            T.ToTensor(),
            # T.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.267, 0.256, 0.276))
        ])
        ds_train = torchvision.datasets.SVHN(root, split="train", transform=transform, download=True)
        ds_eval = torchvision.datasets.SVHN(root, split="test", transform=transform, download=True)
    elif dataset == "caltech256":
        transform = T.Compose([
            # T.HFlip(),
            T.ToTensor(),
            # T.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.267, 0.256, 0.276))
        ])
        ds_train = torchvision.datasets.Caltech256(root, transform=transform, download=True)
        ds_eval = torchvision.datasets.Caltech256(root, transform=transform, download=True)
    elif dataset == "places365":
        ds_train = torchvision.datasets.Places365(
            root=root, 
            split="train-standard", # "train-challenge" "train-standard"
            small=True, 
            download=False, 
            transform=transform_train, 
            target_transform=None, 
        )
        ds_eval = torchvision.datasets.Places365(
            root=root, 
            split="val", 
            small=True, 
            download=False, 
            transform=transform_val, 
            target_transform=None, 
        )
        ds_train = DatasetWrapper(ds_train.imgs, transform=transform_train, name="places365")
        ds_eval = DatasetWrapper(ds_eval.imgs, transform=transform_val, name="places365")
    elif dataset == "inaturalist":
        ds_train = torchvision.datasets.INaturalist(
            root=root, 
            version = '2021_train', 
            target_type = 'full', 
            transform=transform_train, 
            target_transform=None, 
            download=False, 
        )
        ds_eval = torchvision.datasets.INaturalist(
            root=root, 
            version = '2021_valid', 
            target_type = 'full', 
            transform=transform_val, 
            target_transform=None, 
            download=False, 
        )
        
        train_samples_ = [
            (str(Path(ds_train.root, ds_train.all_categories[cat_id], fname)), cat_id)
            for cat_id, fname in ds_train.index
        ]        
        eval_samples_ = [
            (str(Path(ds_eval.root, ds_eval.all_categories[cat_id], fname)), cat_id)
            for cat_id, fname in ds_eval.index
        ]        
        ds_train = DatasetWrapper(train_samples_, transform=transform_train, name="inaturalist")
        ds_eval = DatasetWrapper(eval_samples_, transform=transform_val, name="inaturalist")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    if subsample_classes > 0:
        ds_train = subsample_dataset(ds_train, num_samples=subsample_classes, sample_strategy="sample_classes")
        ds_eval = subsample_dataset(ds_eval, num_samples=subsample_classes, sample_strategy="sample_classes", class_id_map=ds_train.new_class_ids)

    if train_samples > 0:
        ds_train = subsample_dataset(ds_train, train_samples, sample_strategy)
        
    print(f"Train: {len(ds_train)} samples, {len(np.unique(ds_train.targets))} classes.")
    print(f"Eval: {len(ds_eval)} samples, {len(np.unique(ds_eval.targets))} classes.")

    if repeated_aug:
        sampler_train = RepeatAugSamplerV2(
            ds_train, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_global_rank(),
            shuffle=True,
        )
        # sampler_train.__class__ = DistributedSampler
        # sampler_train.seed = 0
        # sampler_train.drop_last = True
        # sampler_train.__iter__ = types.MethodType(RepeatAugSampler.__iter__, sampler_train)
    else:
        sampler_train = dist.get_sampler(ds_train, shuffle=True, drop_last=drop_last_train)
    sampler_eval = dist.get_sampler(ds_eval, shuffle=False)

    train_dataloader = DataLoader(
        ds_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        sampler=sampler_train,
    )
    eval_dataloader = DataLoader(
        ds_eval,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        sampler=sampler_eval,
    )
    
    # print(next(iter(train_dataloader))[0].shape)
    # for i in iter(sampler_train):
    #     print(i)

    return train_dataloader, eval_dataloader
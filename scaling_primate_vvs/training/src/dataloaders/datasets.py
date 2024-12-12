from typing import Union, List, Tuple
from pathlib import Path
from collections import defaultdict
import warnings
import math

import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class ImageFolder_Albumentations(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = np.array(self.loader(path))
        # image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, target
    
class DatasetFileList(Dataset):
    """
    A custom dataset class that loads file paths and target classes from a file list.

    Args:
        root (str or Path): Root directory of the dataset.
        filelist_path (str): Path to the file list.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
            Default is None.

    Attributes:
        root (Path): Root directory of the dataset.
        transform (callable): A function/transform that takes in an image and returns a transformed version.
        samples (list): List of tuples containing the image paths and corresponding targets.
        targets (list): List of targets.

    Methods:
        loader(path): Loads an image from the given path.
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the image and target at the given index.
    """

    def __init__(self, root, file_list_path, transform=None):
        self.root = root if isinstance(root, Path) else Path(root)
        self.transform = transform
        with open(file_list_path, "r") as f:
            file_list = f.readlines()
            self.samples = [(x.split(' ')[0].strip(), int(x.split(' ')[1].strip())) for x in file_list]
            print(f"Loaded {len(self.samples)} samples from {file_list_path}")
        self.targets = [x[1] for x in self.samples]
            
    def loader(self, path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Image.open(self.root / path).convert('RGB')
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = np.array(self.loader(path))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, target
    
class DatasetWrapper(Dataset):
    """
    A custom dataset class to convert various PyTorch datasets to a common format.

    Attributes:
        samples (List[Tuple[str, int]]): A list of tuples where each tuple contains a file path and a target label.
        targets (List[int], optional): A list of target labels. If not provided, it will be extracted from the samples.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
    Methods:
        loader(path):
            Loads an image from the given path and converts it to RGB format.
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(index):
            Retrieves the image and target label at the specified index.
    """

    def __init__(self, samples:List[Tuple[str, int]], targets:List[int]= None, transform=None, name=None):
        self.transform = transform
        self.samples = samples
        if targets is not None:
            self.targets = targets
        else:
            self.targets = [x[1] for x in self.samples]
        self.name = name
            
    def loader(self, path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Image.open(path).convert('RGB')
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = np.array(self.loader(path))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, target
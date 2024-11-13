import sys
import functools

import numpy as np
from PIL import Image
import torchvision.transforms as T
import albumentations as A
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper


if '../' not in sys.path:
    sys.path.append('../')

from training.src.dataloaders.transforms import create_transforms_albumentations



def custom_image_preprocess(images,
                            transforms=None,
                            resize_size: int = 256,
                            crop_size: int = 224,
                            interpolation: str = 'bilinear'
                            ):
    
    if transforms is None:
        _, transforms = create_transforms_albumentations(val_crop_size=crop_size, val_resize_size=resize_size, interpolation=interpolation)
    
    if isinstance(transforms, T.Compose):
        images = [transforms(image) for image in images]
        images = [np.array(image) for image in images]
        images = np.stack(images)
    elif isinstance(transforms, A.Compose):
        images = [np.array(image) for image in images]
        images = [transforms(image=image)["image"] for image in images]
        images = np.stack(images)
    else:
        raise NotImplementedError(f'Transforms of type {type(transforms)} is not implemented')

    return images

def load_preprocess_images_custom(image_filepaths, preprocess_images=custom_image_preprocess,  **kwargs):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = preprocess_images(images, **kwargs)
    return images

def load_image(image_filepath):
    return Image.open(image_filepath).convert('RGB')

def wrap_model(identifier, model, transforms=None, resize_size=256, crop_size=224, interpolation='bilinear'):
    preprocessing = functools.partial(load_preprocess_images_custom, transforms=transforms, resize_size=resize_size, crop_size=crop_size, interpolation=interpolation)
    wrapper = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = crop_size
    return wrapper

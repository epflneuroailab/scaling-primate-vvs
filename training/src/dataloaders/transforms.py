from typing import Union, Literal
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data.auto_augment import rand_augment_transform
from timm.data import create_transform as create_transforms_timm
import lightly.transforms as LT

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_transforms(**kwargs) -> tuple:
    """
    Returns the train and validation transforms based on the specified transform library.

    Args:
        args: The arguments containing the transform library and other parameters.

    Returns:
        A tuple containing the train and validation transforms.
    """
    transform_lib = kwargs.get("transform_lib")
    assert transform_lib in ['albumentations', 'pytorch', 'timm', 'lightly'], \
        f"transform_lib must be one of ['albumentations', 'pytorch', 'timm', 'lightly']"
        
    if transform_lib == "timm":
            train_transform = create_transforms_timm(
                input_size=kwargs.get("train_crop_size"),
                is_training=True,
                color_jitter=kwargs.get("color_jitter"),
                auto_augment=kwargs.get("aa"),
                interpolation=kwargs.get("interpolation"),
                re_prob=kwargs.get("re_prob"),
                re_mode=kwargs.get("re_mode"),
                re_count=kwargs.get("re_count"),
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            )
            _, val_transform = create_transforms_pytorch(
                train_crop_size = kwargs.get("train_crop_size"),
                val_resize_size = kwargs.get("val_resize_size"),
                val_crop_size = kwargs.get("val_crop_size"),
                interpolation = kwargs.get("interpolation"),
                color_jitter = 0,
                aug_set = 'default'
            )
    elif transform_lib == "albumentations":
        train_transform, val_transform = create_transforms_albumentations(    
            train_crop_size = kwargs.get("train_crop_size"),
            val_resize_size = kwargs.get("val_resize_size"),
            val_crop_size = kwargs.get("val_crop_size"),
            interpolation = kwargs.get("interpolation"),
            aug_set = kwargs.get("albumentations_aug_set"),
            rand_aug = kwargs.get("aa"),
            color_jitter= kwargs.get("color_jitter"),
        )
    elif transform_lib == "lightly":
        train_transform, val_transform = create_transforms_lightly(**kwargs)
    else:
        train_transform, val_transform = create_transforms_pytorch(
            train_crop_size = kwargs.get("train_crop_size"),
            val_resize_size = kwargs.get("val_resize_size"),
            val_crop_size = kwargs.get("val_crop_size"),
            interpolation = kwargs.get("interpolation"),
            color_jitter = kwargs.get("color_jitter"),
            aug_set = kwargs.get("pytorch_aug_set"),
        )
    
        
    return train_transform, val_transform
    
    

def create_transforms_pytorch(
    train_crop_size: int = 224,
    val_resize_size: int = 256,
    val_crop_size: int = 224,
    interpolation: str = 'bilinear',
    color_jitter: float = 0.4,
    aug_set: str = 'default'
    ) -> tuple:
    """Returns torchvision transforms for training and validation sets."""
    
    interpolation = get_interpolation_mode(interpolation)
    
    if aug_set == 'default':
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    elif aug_set == 'ThreeAugment':
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.RandomGrayscale(p=1.0),
                    # transforms.RandomSolarize(threshold=128, p=1.0),
                    LT.RandomSolarization(prob=1.0),
                    LT.GaussianBlur(prob=1.0, sigmas=(0.1, 2.0)),
                    ]),
                transforms.ColorJitter(color_jitter, color_jitter, color_jitter),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    elif aug_set == 'kNN':
        transform_train = transforms.Compose(
            [
                transforms.Resize(val_resize_size, interpolation=interpolation),
                transforms.CenterCrop(train_crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        raise NotImplementedError(f'Augmentation set {aug_set} not implemented')

    transform_val = transforms.Compose(
        [
            transforms.Resize(val_resize_size, interpolation=interpolation),
            transforms.CenterCrop(val_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform_train, transform_val


def create_transforms_albumentations(    
    train_crop_size: int = 224,
    val_resize_size: int = 256,
    val_crop_size: int = 224,
    interpolation: str = 'bilinear',
    aug_set: Literal['default', 'heavy', 'ThreeAugment'] = 'default',
    color_jitter: float = 0,
    rand_aug: str = None
) -> tuple:
    """Returns the default albumentations transforms for imagenet"""
    
    interpolation = get_interpolation_mode(interpolation)
    
    transform_train = [
        A.RandomResizedCrop(train_crop_size, train_crop_size, p=1.0, interpolation=interpolation),
        A.HorizontalFlip(p=0.5),
    ]
    
    if aug_set == 'default':
        pass
    elif aug_set == 'heavy':
        transform_train.extend([
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomFog(p=0.1),
            A.ISONoise(p=0.1),
            A.MotionBlur(p=0.1),
            A.ElasticTransform(p=0.1),
            A.RandomRain(p=0.1),
            A.MultiplicativeNoise(p=0.1),
            A.CoarseDropout(p=0.2),
            A.GaussNoise(p=0.1),
        ])
    elif aug_set == 'ThreeAugment':
        transform_train.extend([
            A.OneOf([
                A.ToGray(p=1.0),
                A.GaussianBlur(p=1.0, sigma_limit=(0.1, 2.0)),
                A.Solarize(p=1.0),
            ], p=1.0),
            A.ColorJitter(
                brightness=color_jitter, 
                contrast=color_jitter, 
                saturation=color_jitter, 
                hue=0, 
                always_apply=True, 
                p=1.0
            )
        ])
    else:
        raise NotImplementedError(f'Augmentation set {aug_set} not implemented')
    
    if rand_aug is not None and rand_aug != 'None':
        transform_train.append(RandAugment(rand_aug))
    
    transform_train.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    transform_train.append(ToTensorV2())
    transform_train = A.Compose(transform_train)
    
    transform_val = A.Compose(
        [
            A.Resize(val_resize_size, val_resize_size, p=1.0, interpolation=interpolation),
            A.CenterCrop(val_crop_size, val_crop_size, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    
    
    return transform_train, transform_val

def create_transforms_lightly(**kwargs) -> tuple:
    lightly_aug_set = kwargs.get("lightly_aug_set")
    if lightly_aug_set == 'dino':
        global_crops_scale = kwargs.get('global_crops_scale', (0.4, 1.))
        local_crops_scale = kwargs.get('local_crops_scale', (0.05, 0.4))
        local_crops_number = kwargs.get('local_crops_number', 8)
        local_crop_size = kwargs.get('local_crop_size', 96)
        train_transform = LT.dino_transform.DINOTransform(
            global_crop_size=kwargs.get('train_crop_size'),
            global_crop_scale=global_crops_scale,
            local_crop_size=local_crop_size,
            local_crop_scale=local_crops_scale,
            n_local_views=local_crops_number,
        )
        _, val_transform = create_transforms_pytorch(
            train_crop_size = kwargs.get('train_crop_size'),
            val_resize_size = kwargs.get('val_resize_size'),
            val_crop_size = kwargs.get('val_crop_size'),
            interpolation = kwargs.get('interpolation'),
            color_jitter = 0,
            aug_set = 'default'
        )
    elif lightly_aug_set == 'simclr':
        train_transform = LT.simclr_transform.SimCLRTransform(
            input_size=kwargs.get('train_crop_size'),
        )
        _, val_transform = create_transforms_pytorch(
            train_crop_size = kwargs.get('train_crop_size'),
            val_resize_size = kwargs.get('val_resize_size'),
            val_crop_size = kwargs.get('val_crop_size'),
            interpolation = kwargs.get('interpolation'),
            color_jitter = 0,
            aug_set = 'default'
        )
    
    else:
        raise NotImplementedError(f'Augmentation set {lightly_aug_set} not implemented')
    
    return train_transform, val_transform
    
    


def get_interpolation_mode(interpolation: str) -> int:
    """Returns the interpolation mode for albumentations"""
    if 'linear' or 'bilinear' in interpolation:
        return 1
    elif 'cubic' or 'bicubic' in interpolation:
        return 2
    else:
        raise NotImplementedError(f'Interpolation mode {interpolation} not implemented')


class RandAugment:
    """Wrapper for RandAugment from timm.data.auto_augment"""
    def __init__(self, config_str, hparams={}):
        self.rand_aug = rand_augment_transform(config_str=config_str, hparams=hparams)

    def __call__(self, image, *args, **kwargs):
        return {"image": np.array(self.rand_aug(Image.fromarray(image)))}
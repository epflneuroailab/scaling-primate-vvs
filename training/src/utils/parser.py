import argparse

    
def str2bool(v):
    """
    Converts a string representation of a boolean value to its corresponding boolean value.
    
    Args:
        v (str): The string representation of the boolean value.
        
    Returns:
        bool: The corresponding boolean value.
    """
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_argparser():
    """Get parser for the training script."""
    parser = argparse.ArgumentParser(description="Training CV models")
    ################## Model ##################
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="cornet_s",
        # choices=["cornet_s", "resnet18"],
        help="model architecture: " + " (default: cornet_s)",
    )
    parser.add_argument(
        "--num-classes",
        default=1000,
        type=int,
        metavar="N",
        help="number of classes in dataset (default: 1000)",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--use-timm",
        dest="use_timm",
        action="store_true",
        help="use a timm model",
    )
    parser.add_argument(
        "--use-open-clip",
        dest="use_open_clip",
        action="store_true",
        help="use an open clip model",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "--compile-mode",
        dest="compile_mode",
        default="None",
        type=str,  
        choices=["None", "default", "" "max-autotune", "reduce-overhead"],
        help="compile mode (default: None)",
    )
    parser.add_argument(
        "--channels-last",
        dest="channels_last",
        action="store_true",
        help="Use channels last memory format",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--device", 
        default="gpu", 
        type=str, 
        help="device to use for training / testing"
    )
    parser.add_argument(
        "--ngpus", 
        default=1, 
        type=int, 
        help="number of gpus to use for training / testing. Used for total batch calculation"
    )
    parser.add_argument(
        "--layer-config",
        dest="layer_config",
        type=str,
        default=None,
        help="Layer configuration for ResNetFlex model (default: None)",
    )
    parser.add_argument(
        '--model-ema', 
        dest='model_ema',
        type=str2bool, 
        default=False,
        help='enable model exponential moving average (default: False)'
        )
    parser.add_argument(
        '--model-ema-decay', 
        dest='model_ema_decay',
        type=float, 
        default=0.9999, 
        help='decay factor for model_ema (default: 0.9999)'
        )
    parser.add_argument(
        '--model-ema-eval',
        dest='model_ema_eval',
        type=str2bool,
        default=False,
        help='use model_ema for evaluation (default: False)'
    )
    parser.add_argument(
        '--load-model-ema',
        dest='load_model_ema',
        type=str2bool,
        default=False,
        help='Load model ema weight to model.ema (default: False)'
    )
    parser.add_argument(
        '--ssl-method',
        dest='ssl_method',
        type=str,
        default=None,
        choices=['simclr', 'dino'],
        help='SSL method to use (default: None)'
    )
    parser.add_argument(
        '--sync_batchnorm',
        dest='sync_batchnorm',
        type=str2bool,
        default=False,
        help='Replace batch norm layers with torch.nn.SyncBatchNorm (default: False)'
    )
    parser.add_argument(
        '--precision',
        dest='precision',
        type=str,
        default='amp_fp16',
        choices=['fp32', 'amp_fp16', 'amp_bf16', 'amp_fp8'],
        help='Mixed precision training (default: amp_fp16)'
    )
    parser.add_argument(
        '--adv-config',
        dest='adv_config',
        type=str,
        default=None,
        help='Path to adversarial config file (default: None)'
    )
    ################## Dataset ##################
    parser.add_argument(
        "--data-path", 
        dest="data_path",
        metavar="DIR", 
        help="path to dataset"
    )
    parser.add_argument(
        "--dataset",
        default="imagenet",
        type=str,
        choices=["imagenet", "ecoset", "webvision", "imagenet21k", "imagenet21kP", "webvisionP", "places365", "inaturalist", "infimnist"],
        metavar="DATASET",
        help="dataset to use (default: imagenet)",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=32,
        type=int,
        metavar="N",
        help="mini-batch size (default: 32)",
    )
    parser.add_argument(
        "--target-batch-size",
        dest="target_batch_size",
        default=None,
        type=int,
        metavar="N",
        help="Total batch size for training (default: 64). Based on ngpus, this will be used to calculate" \
                + " the accumulation steps and batch_size will be set as per gpu mini batch size",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--pin-memory", 
        dest="pin_memory",
        default=True, 
        type=bool, 
        help="pin_memory for dataloader"
    )
    parser.add_argument(
        "--train-samples",
        dest="train_samples",
        default=0,
        type=int,
        metavar="N",
        help="number of training samples for subsampling (default: 0)",
    )
    parser.add_argument(
        "--sample-strategy",
        dest="sample_strategy",
        default="per_class",
        type=str,
        choices=["per_class", "whole_dataset"],
        help="sample strategy for subsampling (default: per_class)",
    )
    parser.add_argument(
        "--subsample-classes",
        dest="subsample_classes",
        default=0,
        type=int,
        metavar="N",
        help="number of classes to subsample (default: 0)",
    )
    parser.add_argument(
        "--repeated-aug",
        dest="repeated_aug",
        type=str2bool,
        default=False,
        help="repeated augmentation sampler (default: False)"
    )
    parser.add_argument(
        "--drop-last-train",
        dest="drop_last_train",
        type=str2bool,
        default=None,
        help="Drop last batch in training dataloader. Useful if training dataset is subsampled (default: None)"
    )
    
    ################## Training / Optimizer ##################
    parser.add_argument(
        "--loss-fn",
        dest="loss_fn",
        default="cross_entropy",
        type=str,
        choices=["cross_entropy", "soft_target_cross_entropy", "label_smoothing_cross_entropy", "multiclass_bce", "ntxent_loss", "dino_loss"],
        help="Loss function",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="sgd",
        help="optimizer to use (default: sgd)",
        # choices=["sgd", "adam", "adamw"]
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler",
        default="steplr",
        help="learning rate scheduler (default: steplr)",
        choices=["steplr", "cosineannealinglrwithwarmup", "cosineannealinglr", "constantlr", "plateaulr", "linearwithwarmup"],
    )
    parser.add_argument(
        "--lr-step-size",
        dest="lr_step_size",
        default="30ep",
        type=str,
        metavar="N",
        help="number of epochs to decay learning rate by lr_gamma; ignored if lr_scheduler is cosineannealinglr",
    )
    parser.add_argument(
        "--lr-gamma",
        dest="lr_gamma",
        default=0.1,
        type=float,
        metavar="LR",
        help="multiplicative factor of learning rate decay; ignored if lr_scheduler is cosineannealinglr",
    )
    parser.add_argument(
        "--lr-warmup-duration",
        dest="lr_warmup_duration",
        default='5ep',
        type=str,
        metavar="N",
        help="Duration of learning rate warmup (default: 5ep)",
    )
    parser.add_argument(
        "--min-lr",
        dest="min_lr",
        default=1e-5,
        type=float,
        metavar="LR",
        help="minimum learning rate",
    )
    parser.add_argument(
        "--plateaulr-patience",
        dest="plateaulr_patience",
        default=3,
        type=int,
        metavar="N",
        help="patience for plateau lr scheduler (default: 3)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        '--weight-decay-end',
        dest='weight_decay_end',
        type=float, 
        default=None, 
        help='Weight decay end (default: None)'
    )
    parser.add_argument(
        '--use-weight-decay-scheduler',
        dest='use_weight_decay_scheduler',
        type=str2bool,
        default=False,
        help='Use a cosine scheduler for weight decay (default: False)'
    )
    parser.add_argument(
        '--opt-eps',
        dest='opt_eps',
        default=None, 
        type=float, 
        metavar='EPSILON',
        help='Optimizer Epsilon (default: None, use opt default)'
    )
    parser.add_argument(
        '--opt-betas',
        dest='opt_betas',
        default=None, 
        type=float, 
        nargs='+', 
        metavar='BETA',
        help='Optimizer Betas (default: None, use opt default)'
    )
    parser.add_argument(
        '--clip-grad', 
        dest='clip_grad',
        type=float, 
        default=None, 
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)'
    )
    parser.add_argument(
        '--label-smoothing',
        dest='label_smoothing',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Label smoothing (default: 0.0)'
    )

    
    parser.add_argument(
        "--eval-interval",
        dest="eval_interval",
        default="1ep",
        type=str,
        metavar="N",
        help="evaluate every N epochs (default: 1ep)",
    )
    parser.add_argument(
        "--max-duration",
        dest="max_duration",
        default='100ep',
        type=str,
        metavar="N",
        help="maximum duration for training (default: 90ep)",
    )
    parser.add_argument(
        '--accumulation-steps',
        dest='accumulation_steps', 
        default=1,
        type=int,
        help='gradient accumulation steps'
        )

    
    ################## Data Augmentation ##################
    parser.add_argument(
        '--transform-lib',
        dest='transform_lib',
        type=str,
        default='albumentations',
        choices=['albumentations', 'pytorch', 'timm', 'lightly'],
        help='Library to use for data augmentation (default: albumentations)',
    )
    parser.add_argument(
        "--train-crop-size",
        dest="train_crop_size",
        default=224,
        type=int,
        metavar="N",
        help="crop size for training (default: 224)",
    )
    parser.add_argument(
        "--val-resize-size",
        dest="val_resize_size",
        default=256,
        type=int,
        metavar="N",
        help="resize size for validation (default: 256)",
    )
    parser.add_argument(
        "--val-crop-size",
        dest="val_crop_size",
        default=224,
        type=int,
        metavar="N",
        help="crop size for validation (default: 224)",
    )
    parser.add_argument(
        '--interpolation',
        default='bilinear',
        type=str,
        metavar="N",
        help="interpolation mode for resizing (default: bilinear)",
    )
    parser.add_argument(
        '--albumentations-aug-set',
        dest='albumentations_aug_set',
        type=str,
        default="default",
        choices=['default', 'heavy', 'ThreeAugment'],
        help='Albumentations augmentation set (default: default)',
    )
    parser.add_argument(
        '--lightly-aug-set',
        dest='lightly_aug_set',
        type=str,
        default=None,
        choices=['dino', 'simclr'],
        help='Lightly augmentation set for SSL training (default: None)',
    )
    parser.add_argument(
        '--pytorch-aug-set',
        dest='pytorch_aug_set',
        type=str,
        default=None,
        choices=['default', 'ThreeAugment', 'kNN'],
        help='Pytorch augmentation set(default: None)',
    )
    parser.add_argument(
        '--color-jitter',
        dest='color_jitter', 
        type=float, 
        default=0.0, 
        metavar='PCT',
        help='Color jitter factor (default: 0.0)'
    )
    parser.add_argument(
        '--aa', 
        type=str, 
        # default='rand-m9-mstd0.5-inc1',
        default=None,
        metavar='NAME',
        help='Use AutoAugment policy. "rand-m9-mstd0.5-inc1". " + "(default: None)'
    )
    parser.add_argument(
        '--re-prob', 
        dest='re_prob',
        type=float, 
        default=0.0, 
        metavar='PCT',
        help='Random erase probability (default: 0.0)'
    )
    parser.add_argument(
        '--re-mode',
        dest='re_mode',
        type=str, 
        default='pixel',
        help='Random erase mode (default: "pixel")'
    )
    parser.add_argument(
        '--re-count',
        dest='re_count',
        type=int, 
        default=1,
        help='Random erase count (default: 1)'
    )
    
    ################## Mixup/Cutmix ##################
    parser.add_argument(
        '--mixup', 
        type=float, 
        default=0.0,
        help='MixUp probability (alpha), enabled if > 0.'
    )
    parser.add_argument(
        '--cutmix', 
        type=float, 
        default=0.0,
        help='CutMix probability (alpha), enabled if > 0.'
    )
    parser.add_argument(
        '--cutmix-minmax', 
        dest='cutmix_minmax',
        type=float, 
        nargs='+', 
        default=None,
        help='CutMix min/max ratio, overrides alpha and enables cutmix if set (default: None)'
    )
    parser.add_argument(
        '--mixup-prob',
        dest='mixup_prob',
        type=float, 
        default=0.0,
        help='Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup-switch-prob', 
        dest='mixup_switch_prob',
        type=float, 
        default=0.0,
        help='Probability of switching to CutMix when both MixUp and CutMix enabled'
    )
    parser.add_argument(
        '--mixup-mode',
        dest='mixup_mode',
        type=str, 
        default='batch',
        help='How to apply MixUp/CutMix params. Per "batch", "pair", or "elem"'
    )
    
    ################## Logging ##################
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        default="./outputs",
        type=str,
        metavar="PATH",
        help="path to save output (default: ./outputs/)",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default=None,
        type=str,
        help="Run name",
    )
    parser.add_argument(
        "--disable-wandb",
        dest="disable_wandb",
        action="store_true",
        help="Disable wandb for logging",
    )
    parser.add_argument(
        "--wandb-project",
        dest="wandb_project",
        default=None,
        type=str,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        dest="wandb_entity",
        default=None,
        type=str,
        help="Weights & Biases entity name",
    )
    parser.add_argument(
        "--wandb-id",
        dest="wandb_id",
        default=None,
        type=str,
        help="Weights & Biases run id",
    )
    parser.add_argument(
        "--log-dir",
        dest="log_dir",
        default=None,
        type=str,
        help="Log directory for txt files",
    )
    parser.add_argument(
        "--log-interval",
        dest="log_interval",
        default="1ba",
        type=str,
        help="Log interval for console",
    )
    parser.add_argument(
        "--lr-monitor",
        dest="lr_monitor",
        action="store_true",
        help="Monitor LR",
    )
    parser.add_argument(
        "--disable-progress-bar",
        dest="disable_progress_bar",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--save-overwrite",
        dest="save_overwrite",
        action="store_true",
        help="Overwrite save directory",
    )
    return parser

import argparse
import gc
from pathlib import Path
import copy
import yaml

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from composer import Trainer
from composer.algorithms import ChannelsLast, GradientClipping
from composer.loggers import WandBLogger, FileLogger
from composer.utils import reproducibility
from composer.optim import CosineAnnealingWithWarmupScheduler, CosineAnnealingScheduler, ConstantScheduler, StepScheduler, LinearWithWarmupScheduler
from composer.callbacks import LRMonitor
from composer.core import Evaluator, DataSpec

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy

from .models import create_model
from .models.ssl import SSLModelHooks
from .dataloaders import create_dataloaders
from .utils import create_argparser, create_lr_scheduler, create_trainer_algorithms
from .optim import create_optimizer


# from composer.callbacks import EarlyStopper

# torch.backends.cudnn.enabled = False

# torch.backends.cudnn.benchmark = True


def main(args):
    # reproducibility.configure_deterministic_mode()
    reproducibility.seed_all(args.seed)
    config = copy.deepcopy(args)
    
    if args.loss_fn == "cross_entropy":
        assert args.mixup == 0. and args.cutmix == 0., \
            "Mixup and Cutmix not supported with Cross Entropy loss"
        loss_fn = torch.nn.CrossEntropyLoss()
    elif args.loss_fn  == "soft_target_cross_entropy":
        loss_fn = SoftTargetCrossEntropy()
    elif args.loss_fn  == "label_smoothing_cross_entropy" and args.label_smoothing > 0.:
        assert args.mixup == 0. and args.cutmix == 0., \
            "Mixup and Cutmix not supported with Label Smoothing Cross Entropy loss"
        loss_fn = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    elif args.loss_fn == "multiclass_bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.loss_fn in ["dino_loss", "ntxent_loss"]:
        # Loss function is defined in the SSL model
        loss_fn = args.loss_fn
    else:
        raise NotImplementedError(f"Loss function {args.loss_fn} not implemented")
    
    
    args.loss_fn = loss_fn

    model = create_model(**vars(args))
    if args.checkpoint and 'http' in getattr(args, 'checkpoint', ''):
        # Remove the remote checkpoint from the args, otherwise composer will complain
        args.checkpoint = None
    
    if args.dataset == "imagenet":
        assert args.num_classes == 1000, "num_classes must be 1000 for ImageNet"
    elif args.dataset == "ecoset":
        assert args.num_classes == 565, "num_classes must be 565 for ecoset"
    elif args.dataset == "imagenet21k":
        assert args.num_classes == 21841, "num_classes must be 21841 for ImageNet21k (Fall 2011)"
    elif args.dataset == "imagenet21kP":
        assert args.num_classes == 10450, "num_classes must be 10450 for ImageNet21k"
    elif args.dataset == "webvision":
        assert args.num_classes == 5000, "num_classes must be 5000 for WebVision"
    elif args.dataset == "webvisionP":
        assert args.num_classes == 4186, "num_classes must be 4186 for WebVision-P"
    elif args.dataset == "places365":
        assert args.num_classes == 365, "num_classes must be 365 for places365"
    elif args.dataset == "inaturalist":
        assert args.num_classes == 10000, "num_classes must be 10000 for inaturalist"
    
    if args.target_batch_size is not None:
        # Composer.Trainer copies the dataloader across gpus
        # so we need to adjust the batch size accordingly
        total_batch_size = args.target_batch_size
        device_train_microbatch_size = args.batch_size
        args.batch_size = args.target_batch_size // args.ngpus
        args.accumulation_steps = args.target_batch_size // (device_train_microbatch_size * args.ngpus)
    else:
        total_batch_size = args.batch_size * args.ngpus
        device_train_microbatch_size = args.batch_size // args.accumulation_steps
        
    print("Creating dataloaders")
    train_dataloader, eval_dataloader = create_dataloaders(**vars(args))
    num_training_steps_per_epoch = len(train_dataloader.dataset) // total_batch_size
    eval_dataloader = Evaluator(
        label='eval', 
        dataloader=eval_dataloader, 
        device_eval_microbatch_size=device_train_microbatch_size
    )
    print(f"Total batch size = {total_batch_size}")
    print(f"Accumulation steps = {args.accumulation_steps}")
    print(f"Number of training examples = {len(train_dataloader.dataset)}")
    print(f"Number of training samples per epoch = {num_training_steps_per_epoch}")
    
    # Save old class ids if using subsampling of classes
    if hasattr(train_dataloader.dataset, 'old_class_ids'):
        setattr(model, 'new2oldClassMap', train_dataloader.dataset.old_class_ids)
        setattr(model, 'n_train_samples', len(train_dataloader.dataset))
    setattr(args, 'n_train_samples', len(train_dataloader.dataset))
    if args.ssl_method:
        train_dataloader = DataSpec(
            dataloader=train_dataloader,
            get_num_samples_in_batch=model.get_num_samples_in_batch,
            split_batch=model.split_batch,
        )
        eval_dataloader = None
        args.eval_interval = 0

    trainer_algorithms = create_trainer_algorithms(model=model, num_training_steps_per_epoch=num_training_steps_per_epoch, **vars(args))
    ## SSL hooks
    if args.ssl_method:
        ssl_hooks = SSLModelHooks(
            hooks = model.get_hooks(),
        )
        trainer_algorithms.append(ssl_hooks)

    optimizer = create_optimizer(args, model)
    
    schedulers = [create_lr_scheduler(**vars(args), optimizer=optimizer)]

    # Loggers
    loggers = []
    if not args.disable_wandb:
        wandb_id = args.wandb_id if args.wandb_id is not None else args.run_name
        wandb_id = None
        wandb_id = args.wandb_id if args.wandb_id is not None else None
        wandb_logger = WandBLogger(
            project=args.wandb_project, 
            name=args.run_name, 
            entity=args.wandb_entity, 
            init_kwargs={
                'config': config, 
                'resume':'allow', 'id': wandb_id
                }
        )
        loggers.append(wandb_logger)
    if args.log_dir is not None:
        file_logger = FileLogger(
            filename=str(Path(args.log_dir) / args.dataset / args.arch / "{run_name}.txt"),
            buffer_size=1,
            flush_interval=100
        )
        loggers.append(file_logger)
        
    # Callbacks
    callbacks = []
    if args.lr_monitor:
        callbacks.append(LRMonitor())

    save_dir = Path(args.save_dir)
    save_dir = str(save_dir / args.run_name if args.run_name is not None else save_dir)
    
    if args.compile_mode == "None":
        compile_config = None
    else:
        compile_config = {'mode': args.compile_mode}
        
    # compile_config = None
    # if args.compile_mode == "max-autotune":
    #     model = torch.compile(model, mode='max-autotune')
    # elif args.compile_mode == "default":
    #     model = torch.compile(model, mode='default')

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=optimizer,
        schedulers=schedulers,
        max_duration=args.max_duration,
        device=args.device,
        save_folder=save_dir,
        save_overwrite=args.save_overwrite,
        loggers=loggers,
        run_name=args.run_name,
        progress_bar=not args.disable_progress_bar,
        seed=args.seed,
        eval_interval=args.eval_interval,
        callbacks=callbacks,
        load_path=args.checkpoint,
        # deepspeed_config={},
        algorithms=trainer_algorithms,
        compile_config=compile_config,
        console_log_interval=args.log_interval,
        device_train_microbatch_size=device_train_microbatch_size,
        precision=args.precision,
        spin_dataloaders=False,
    )

    trainer.fit()


if __name__ == "__main__":
    
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    
    parser = create_argparser()
    
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        default_config = cfg.get('defaults', None)
        if default_config is not None:
            # Load a base configuration files if exists
            with open(default_config, 'r') as f:
                defaults = yaml.safe_load(f)
            # Update the defaults with the current configuration
            cfg.update({
                k: v for k, v in defaults.items() if k not in cfg
            })
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    
    if args.adv_config:
        with open(args.adv_config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
        args = parser.parse_args(remaining)
        
    
    print(args)

    main(args)

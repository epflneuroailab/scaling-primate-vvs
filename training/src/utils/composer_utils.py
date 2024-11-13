from typing import Any, Callable, List, Tuple, Union
import copy

import numpy as np
import torch

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.algorithms import ChannelsLast, GradientClipping
from timm.data.mixup import Mixup
from timm.utils import ModelEmaV3, get_state_dict

from .schedulers import cosine_schedule_with_warmup_values

class MixUpCutMixTIMM(Algorithm):
    """
    MixUpCutMixTIMM is an algorithm class that applies MixUp and CutMix data augmentation techniques to the input data.

    Args:
        mixup_alpha (float): The alpha parameter for MixUp. Default is 1.0.
        cutmix_alpha (float): The alpha parameter for CutMix. Default is 0.0.
        cutmix_minmax (Any | None): The min-max range for CutMix. Default is None.
        prob (float): The probability of applying MixUp or CutMix. Default is 1.
        switch_prob (float): The probability of switching between MixUp and CutMix. Default is 0.5.
        mode (str): The mode for applying MixUp or CutMix. Default is 'batch'.
        correct_lam (bool): Whether to correct the lambda value for MixUp or CutMix. Default is True.
        label_smoothing (float): The label smoothing factor. Default is 0.1.
        num_classes (int): The number of classes. Default is 1000.
        input_key (Union[str, int, Tuple[Callable, Callable], Any]): The key for accessing the input data. Default is 0.
        target_key (Union[str, int, Tuple[Callable, Callable], Any]): The key for accessing the target data. Default is 1.
    """

    def __init__(
        self,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 0.0,
        cutmix_minmax: Any | None = None,
        prob: float = 1,
        switch_prob: float = 0.5,
        mode: str = 'batch',
        correct_lam: bool = True,
        label_smoothing: float = 0.1,
        num_classes: int = 1000,
        log: bool = False,
        input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
    ):
        self.mixup_fn = Mixup(
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmix_minmax=cutmix_minmax,
            prob=prob, switch_prob=switch_prob, mode=mode, correct_lam=correct_lam,
            label_smoothing=label_smoothing, num_classes=num_classes)

        self.log = log
        self.input_key, self.target_key = input_key, target_key

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.BEFORE_FORWARD, Event.BEFORE_LOSS]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        input, target = state.batch_get_item(key=self.input_key), state.batch_get_item(key=self.target_key)

        if event == Event.BEFORE_FORWARD:
            # Apply the mixup/cutmix
            new_input, new_target = self.mixup_fn(input, target)
            state.batch_set_item(self.input_key, new_input)
            state.batch_set_item(self.target_key, new_target)
            if self.log and state.timestamp.batch_in_epoch.value == 0:
                # Log the images for the first batch in the epoch
                logger.log_images(
                    images=new_input, 
                    name="Mixup/Cutmix images", 
                    step=state.timestamp.batch.value,
                    channels_last=False
                )




class WDCosineWithWarmUp(Algorithm):
    """
    A weight decay algorithm that applies cosine annealing with warm-up.

    Args:
        base_wd (float): The base weight decay value.
        final_wd (float): The final weight decay value.
        epochs (int): The total number of epochs.
        niter_per_ep (int): The number of iterations per epoch.
        warmup_epochs (int, optional): The number of warm-up epochs. Defaults to 0.
        start_warmup_value (float, optional): The initial warm-up value. Defaults to 0.
        log (bool, optional): Whether to log the weight decay values. Defaults to False.
    """

    def __init__(
        self,
        base_wd: float,
        final_wd: float,
        epochs: int, 
        niter_per_ep: int, 
        warmup_epochs: int = 0,
        start_warmup_value: float = 0,
        log: bool = False
    ):
        self.base_wd = base_wd
        self.final_wd = final_wd
        self.epochs = epochs
        self.niter_per_ep = niter_per_ep
        self.warmup_epochs = warmup_epochs
        self.start_warmup_value = start_warmup_value
        self.log = log
        if self.final_wd is None:
            self.final_wd = self.base_wd
        
        self.wd_schedule_values = cosine_schedule_with_warmup_values(
            base_value=self.base_wd, 
            final_value=self.final_wd, 
            epochs=self.epochs, 
            niter_per_ep=self.niter_per_ep, 
            warmup_epochs=self.warmup_epochs,
            start_warmup_value=self.start_warmup_value
        )
        # print(self.wd_schedule_values, np.unique(self.wd_schedule_values))

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.BEFORE_FORWARD]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.BEFORE_FORWARD:
            iter = state.timestamp.batch.value
            assert iter == (state.timestamp.epoch.value * self.niter_per_ep) + state.timestamp.batch_in_epoch.value
            for optim in state.optimizers:
                for param_group in optim.param_groups:

                    if param_group["weight_decay"] > 0:
                        # print(param_group["weight_decay"])
                        param_group["weight_decay"] = self.wd_schedule_values[iter]
                        # print(param_group["weight_decay"])
                        
            if self.log:
                logger.log_metrics(metrics={"weight_decay": self.wd_schedule_values[iter]}, step=iter)
                
                
class EmaWithEvalTIMM(Algorithm):

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        device: str = 'cpu',
        model_ema_eval: bool = False,
        checkpoint_path: str = None,
        metric_funcs: dict = {},
        log: bool = False
    ):
        self.decay = decay
        self.device = device
        self.device = 'cuda:0' if self.device == 'gpu' else self.device
        # self.device = torch.device(self.device)
        self.model_ema_eval = model_ema_eval
        self.checkpoint_path = checkpoint_path
        self.metrics_names = []
        self.metric_funcs = {}
        # self.metric_funcs = metric_funcs
        # self.metrics_names = list(self.metric_funcs.keys())
        
        
        self.model_ema = ModelEmaV3(
            model,
            decay=self.decay,
            device=self.device,
            )
        
        print(f"Using EMA with decay = {self.decay:.8f}")
        
    def get_ema_device(self):
        return next(self.model_ema.parameters()).device
    
    def get_model_device(self, state):
        return next(state.model.parameters()).device


    def match(self, event: Event, state: State) -> bool:
        return event in [
            Event.AFTER_LOAD,
            Event.FIT_START, 
            Event.EVAL_BEFORE_ALL, 
            Event.BATCH_END, 
            Event.EVAL_START, 
            Event.EVAL_AFTER_FORWARD, 
            Event.EVAL_END
            ]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        
        if event == Event.AFTER_LOAD:
            if self.checkpoint_path:
                state_dict = torch.load(self.checkpoint_path, map_location="cpu")
                self.model_ema.module.load_state_dict(state_dict["state"]["model_ema_state_dict"], strict=True)
                print(f"\nLoaded EMA model from {self.checkpoint_path}")
        
        if event == Event.FIT_START:
            setattr(state, "model_ema", state.device.module_to_device(self.model_ema))
        
        if event == Event.BATCH_END:
            model_ema = getattr(state, "model_ema")
            # ema_device, model_device = self.get_ema_device(), self.get_model_device(state)
            # if ema_device == model_device:
                # Only update on the first device
            model_ema.update(state.model)
            # setattr(state, "model_ema", self.model_ema)
        
        # # Evaluate using EMA model
        if self.model_ema_eval:
            # Initialize metrics
            if event == Event.EVAL_BEFORE_ALL:
                ema_metrics = {}
                for evaluator in state.evaluators:
                    for metric_name, metric_func in state.eval_metrics[evaluator.label].items():
                        metric_name = f"metrics/{evaluator.label}/EMA_{metric_name}"
                        metric_func = copy.deepcopy(metric_func) #.to(ema_device)
                        metric_func = state.device.module_to_device(metric_func)
                        metric_func.reset()
                        ema_metrics[metric_name] = metric_func
                setattr(state, "ema_metrics", ema_metrics)
                    
                        
            # Update metrics
            if event == Event.EVAL_AFTER_FORWARD:
                input, target = state.batch_get_item(key=0), state.batch_get_item(key=1)
                preds = state.model_ema.module((input, target))
                
                eval_metrics = getattr(state, "ema_metrics")
                assert isinstance(eval_metrics, dict)
                
                for metric_name, metric_func in eval_metrics.items():
                    metric_func.update(preds, target)
                setattr(state, "ema_metrics", eval_metrics)
                
            # Log metrics
            if event == Event.EVAL_END:
                    
                eval_metrics = getattr(state, "ema_metrics")
                # eval_metrics = copy.deepcopy(getattr(state, "ema_metrics"))
                assert isinstance(eval_metrics, dict)
                
                for metric_name, metric_func in eval_metrics.items():
                    metric_value = metric_func.compute()
                    metric_func.reset()
                    logger.log_metrics(metrics={metric_name: metric_value}, step=state.timestamp.batch.value)
                    
        if event == Event.EVAL_END:
            setattr(state, "model_ema_state_dict", get_state_dict(self.model_ema))
            if 'model_ema' not in state.serialized_attributes:
                state.serialized_attributes.append('model_ema_state_dict')
            
            
def create_trainer_algorithms(**kwargs):
    trainer_algorithms = kwargs.get("trainer_algorithms", [])
    
    ## Channels last
    if kwargs.get('channels_last', False):
        channels_last = ChannelsLast()
        trainer_algorithms.append(channels_last)
        
    ## EMA
    if kwargs.get('model_ema', False):
        model = kwargs.get('model')
        model_ema = EmaWithEvalTIMM(
            model = model,
            decay = kwargs.get('model_ema_decay'),
            device = kwargs.get('device'),
            model_ema_eval = kwargs.get('model_ema_eval'),
            checkpoint_path= kwargs.get('checkpoint'),
            metric_funcs = model.get_metrics(is_train=False)
        )
        trainer_algorithms.append(model_ema)
        
        
    ## Weight decay scheduler
    if kwargs.get('use_weight_decay_scheduler', False):
        wd_scheduler = WDCosineWithWarmUp(
            base_wd=kwargs.get('weight_decay'),
            final_wd=kwargs.get('weight_decay_end'),
            epochs=int(kwargs.get('max_duration').replace('ep', '')),
            niter_per_ep=kwargs.get('num_training_steps_per_epoch'),
            log=kwargs.get('lr_monitor'),
            )
        trainer_algorithms.append(wd_scheduler)
        
    ## Mixup and CutMix
    mixup = kwargs.get('mixup')
    cutmix = kwargs.get('cutmix')
    if mixup or cutmix:
        mixup_cutmix = MixUpCutMixTIMM(
            mixup_alpha=mixup,
            cutmix_alpha=cutmix,
            cutmix_minmax=kwargs.get('cutmix_minmax'),
            prob=kwargs.get('mixup_prob'),
            switch_prob=kwargs.get('mixup_switch_prob'),
            mode=kwargs.get('mixup_mode'),
            label_smoothing=kwargs.get('label_smoothing'),
            num_classes=kwargs.get('num_classes'),
            # log=True # For debugging
        )
        trainer_algorithms.append(mixup_cutmix)
        
    ## Gradient clipping
    clip_grad = kwargs.get('clip_grad')
    if clip_grad is not None and clip_grad > 0.0:
        gradient_clipping = GradientClipping(
            clipping_threshold=clip_grad,
            clipping_type='norm',
        )
        trainer_algorithms.append(gradient_clipping)
        
    return trainer_algorithms
        
    
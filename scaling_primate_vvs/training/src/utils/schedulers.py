import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from composer.optim import CosineAnnealingWithWarmupScheduler, CosineAnnealingScheduler, ConstantScheduler, StepScheduler, LinearWithWarmupScheduler

import numpy as np


def cosine_schedule_with_warmup_values(
    base_value: float, 
    final_value: float, 
    epochs: int, 
    niter_per_ep: int, 
    warmup_epochs:int = 0,
    start_warmup_value: float = 0
    ):
    """
    Generate a cosine schedule with warmup.

    Args:
        base_value (float): The base value.
        final_value (float): The final value.
        epochs (int): The total number of epochs.
        niter_per_ep (int): The number of iterations per epoch.
        warmup_epochs (int, optional): The number of warmup epochs. Defaults to 0.
        start_warmup_value (float, optional): The initial value for warmup. Defaults to 0.

    Returns:
        np.ndarray: The value schedule.
    """
    
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    else:
        warmup_schedule = np.array([])

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


class ReduceLROnPlateauScheduler:
    """Wrapper of torch.optim.lr_scheduler.ReduceLROnPlateau with __call__ method to use with
       Composer's Trainer.
       ref: https://github.com/pytorch/ignite/issues/462#issuecomment-771907302
       
       !!! CURRENTLY NOT WORKING !!!
    """

    def __init__(
            self,
            optimizer,
            metric_name,
            mode='min', factor=0.1, patience=10,
            threshold=1e-4, threshold_mode='rel', cooldown=0,
            min_lr=0, eps=1e-8, verbose=False,
    ):
        self.metric_name = metric_name
        self.scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                           threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                                           min_lr=min_lr, eps=eps, verbose=verbose)

    def __call__(self,state):
        
        metric = state.eval_metric_values.get(self.metric_name, 0)
        
        self.scheduler.step(metric)

    def state_dict(self):
        return self.scheduler.state_dict()
    


def create_lr_scheduler(**kwargs):
    lr_scheduler = kwargs.get("lr_scheduler")
    
    if lr_scheduler == "cosineannealinglrwithwarmup":
        scheduler = CosineAnnealingWithWarmupScheduler(
            t_warmup=kwargs.get("lr_warmup_duration"),
            t_max=kwargs.get("max_duration"),
            alpha_f=kwargs.get("min_lr")/kwargs.get("lr")
        )
        return scheduler
    elif lr_scheduler == "linearwithwarmup":
        scheduler = LinearWithWarmupScheduler(
            t_warmup=kwargs.get("lr_warmup_duration"),
            t_max=kwargs.get("max_duration"),
            alpha_f=kwargs.get("min_lr")/kwargs.get("lr")
        )
        return scheduler
    elif lr_scheduler == "constantlr":
        scheduler = ConstantScheduler(
            t_warmup=kwargs.get("lr_warmup_duration"),
            alpha=1
        )
    elif lr_scheduler == "cosineannealinglr":
        scheduler = CosineAnnealingScheduler(
            t_max=kwargs.get("max_duration"),
            alpha_f=kwargs.get("min_lr")/kwargs.get("lr")
        )
        return scheduler
    elif lr_scheduler == "steplr":
        scheduler = StepScheduler(
            step_size=kwargs.get("lr_step_size"),
            gamma=kwargs.get("lr_gamma")
        )
        return scheduler
    elif lr_scheduler == "plateaulr":
        raise NotImplementedError
        scheduler = ReduceLROnPlateauScheduler(
            optimizer=kwargs.get("optimizer"), 
            metric_name='val_loss',
            mode='min', 
            factor=kwargs.get("lr_gamma"), 
            patience=kwargs.get("plateaulr_patience"), 
            verbose=True, 
            threshold=0.0001, 
            threshold_mode='rel', 
            cooldown=0, 
            min_lr=kwargs.get("min_lr"), 
            eps=1e-08
        )
        return scheduler
    else:
        raise NotImplementedError
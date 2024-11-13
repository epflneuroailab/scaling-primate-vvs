from typing import Any, Tuple

import torch
import torchvision
from torch import nn

from composer.models import ComposerClassifier
from composer.metrics import CrossEntropy

import torchmetrics

import torchattacks

class AdvModel(ComposerClassifier):
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]
    
    def __init__(
            self,
            module: torch.nn.Module,
            num_classes: int,
            attack_mode: str, 
            eps: float,
            alpha: float,
            steps: int,
            *args, **kwargs
        ) -> None:
        """
        Initialize the AdversarialAttack class.
        
        Parameters:
            module (torch.nn.Module): The model to be attacked.
            attack_mode (str): The type of attack to be used. Supported options are 'ffgsm', 'pgd_linf', and 'pgd_l2'.
            eps (float): The maximum perturbation allowed for the attack.
            alpha (float): The step size for generating adversarial examples.
            steps (int): The number of steps to perform during a attack.
            *args, **kwargs: Additional arguments to be passed to the superclass constructor.
        
        Raises:
            ValueError: If the specified attack_mode is not supported.
        """
        
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        
        # Loss function
        loss_fn = kwargs.get("loss_fn", None)
        if loss_fn is None or isinstance(loss_fn, nn.CrossEntropyLoss()):
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function")
            
    
        # Metrics
        val_metrics = kwargs.get("val_metrics", None)
        if val_metrics is None:
            val_metrics = {}
            val_metrics["MulticlassAccuracy"] = torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes, average="micro"
            )
            val_metrics["CrossEntropyLoss"] = CrossEntropy()
        val_metrics.update({
            f"{k}_adv": v.clone() for k, v in val_metrics.items()
        })
        for k, v in val_metrics.items():
            if '_adv' in k:
                setattr(v, 'is_adv', True)
            else:
                setattr(v, 'is_adv', False)
        val_metrics = torchmetrics.MetricCollection(val_metrics)
        
        super().__init__(module=module, num_classes=num_classes, val_metrics=val_metrics, loss_fn=loss_fn, *args, **kwargs)
        
        if attack_mode == 'ffgsm':
            print(f"Using FFGSM attack with eps={self.eps}, alpha={self.alpha}")
            self.attack_model = torchattacks.FFGSM(self.module, eps=self.eps, alpha=self.alpha)
        elif attack_mode == 'pgd_linf':
            print(f"Using PGD-Linf attack with eps={self.eps}, alpha={self.alpha}, steps={self.steps}")
            self.attack_model = torchattacks.PGD(self.module, eps=self.eps, alpha=self.alpha, steps=self.steps)
        elif attack_mode == 'pgd_l2':
            print(f"Using PGD-L2 attack with eps={self.eps}, alpha={self.alpha}, steps={self.steps}")
            self.attack_model = torchattacks.PGDL2(self.module, eps=self.eps, alpha=self.alpha, steps=self.steps)
        else:
            raise ValueError(f"Unsupported attack type {attack_mode}")
            
        self.attack_model.set_normalization_used(
            mean=self.normalization_mean,
            std=self.normalization_std,
        )
        
    # def forward(self, batch: Any) -> torch.Tensor | Sequence[torch.Tensor]:
    def forward(self, batch: Tuple[torch.Tensor, Any]) -> torch.Tensor:
        images, labels, adv_images, outputs = self._forward(batch)
        return outputs
    
    def _forward(self, batch: Tuple[torch.Tensor, Any]):
        images, labels = batch
        self.attack_model.set_device(images.device)
        with torch.enable_grad():
            adv_images = self.attack_model(images, labels)
        outputs = self.module(adv_images)
        return images, labels, adv_images, outputs
    
    # def eval_forward(self, batch: Tuple[torch.Tensor, Any], outputs: Any | None = None) -> Any:
    #     images, labels, adv_images, outputs = self._eval_forward(batch)
    #     return outputs
    
    # def _eval_forward(self, batch: Tuple[torch.Tensor, Any]):
    #     images, labels = batch
    #     self.attack_model.set_device(images.device)
    #     # torch.set_grad_enabled(True)  # Context-manager 
    #     # self.attack_model.model.train()
    #     with torch.enable_grad():
    #         print(images.requires_grad)
    #         images.requires_grad = True
    #         adv_images = self.attack_model(images, labels)
    #     # torch.set_grad_enabled(False)  # Context-manager 
    #     outputs = self.module(adv_images)
    #     return images, labels, adv_images, outputs
    
    # # def forward(self, batch: Any) -> torch.Tensor | Sequence[torch.Tensor]:
    # def eval_forward(self, batch: Tuple[torch.Tensor, Any]) -> torch.Tensor:
    #     images, labels, adv_images, outputs = self._forward(batch)
    #     return outputs
    
    def get_metrics(self, is_train=False):
        return {} if is_train else self.val_metrics
    
    def loss(self, outputs: torch.Tensor, batch: Tuple[torch.Tensor, Any], **kwargs) -> torch.Tensor:
        _, targets = batch        
        loss = self._loss_fn(outputs, targets)
        return loss
            
            
    def update_metric(self, batch: Tuple[torch.Tensor, Any], outputs: torch.Tensor, metric: torchmetrics.Metric) -> None:
        # metric_fn_name = metric.__class__.__name__
        inputs, targets = batch
        
        if metric.is_adv:
            metric.update(outputs, targets)
        else:
            # Outputs are the adversarial outputs
            # We need to compute the loss using the original inputs
            outputs = self.module(inputs)
            metric.update(outputs, targets)
                
def wrap_adv_model(model: ComposerClassifier, **kwargs) -> AdvModel:
    base_model = model.module
    
    attack_mode = kwargs.get('adv_attack_mode', None)
    epsilon = float(kwargs.get('adv_eps', 8/255))
    alpha = float(kwargs.get('adv_alpha', 10/255))
    divisior = float(kwargs.get('adv_divisor', 1))
    steps = int(kwargs.get('adv_steps', 10))
    
    epsilon = epsilon / divisior
    alpha = alpha / divisior
    
    assert attack_mode in ['ffgsm', 'pgd_linf', 'pgd_l2'], f"Unsupported attack type {attack_mode}"
    
    return AdvModel(
        module=base_model, 
        attack_mode=attack_mode, 
        eps=epsilon, 
        alpha=alpha, 
        steps=steps,
        num_classes=kwargs.get('num_classes'),
    )
import copy
from typing import Literal, Sequence, Any, Tuple, List, Dict

import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss, DINOLoss
from lightly.models.modules import SimCLRProjectionHead, DINOProjectionHead
from lightly.models.utils import update_momentum
from lightly.models import utils as lightly_utils
from lightly.utils.scheduler import cosine_schedule

from composer.core import Event, State, Algorithm
from composer.loggers import Logger
from composer import ComposerModel
from composer.utils import dist
import timm

from .deit import replace_drop_path
        

class SSLBase(ComposerModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.current_epoch = 0
        self.current_step = 1
        self.hooks = {
            Event.EPOCH_END: [self.update_epoch],
            Event.BATCH_END: [self.update_step]
        }
        self.forward = self._forward
        
    def eval(self):
        self.forward = self.extract_features
        return super().train(mode=False)
        
    def train(self, mode: bool = True):
        self.forward = self._forward
        return super().train(mode)
        
    # def forward(self, batch: Any) -> torch.Tensor | Sequence[torch.Tensor]:
    def forward(self, batch: Any):
        NotImplementedError
        
    def _forward(self, batch: Any):
        NotImplementedError
    
    # def eval_forward(self, batch: Any) -> torch.Tensor | Sequence[torch.Tensor]:
    def eval_forward(self, batch: Any):
        raise NotImplementedError
    
    def extract_features(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def get_metrics(self, is_train=False):
        return {}
    
    def get_hooks(self):
        return self.hooks
    
    def update_epoch(self):
        self.current_epoch += 1
        
    def update_step(self):
        self.current_step += 1
        
    @staticmethod
    def get_num_samples_in_batch(batch)->int:
        # 2 x n_views x batch_size x channels x height x width
        return len(batch[0][0])
    
    @staticmethod
    def split_batch(batch, microbatch_size):
        inputs, targets = batch
        microbatches = []
        for start in range(0, len(inputs[0]), microbatch_size):
            microbatch = []
            for view in inputs:
                microbatch.append(view[start:start+microbatch_size])
            microbatches.append(microbatch)
        microbatches = (microbatches, [targets]*len(microbatches))
        return list(zip(*microbatches))
    
class DINOProjectionHeadV2(DINOProjectionHead):
    """DINO projection head with updated normal std for initialization."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def _init_weights(self, module: nn.Module) -> None:
        """Initializes layers with a truncated normal distribution, std is set to DINO original value."""
        if isinstance(module, nn.Linear):
            lightly_utils._no_grad_trunc_normal(
                module.weight,
                mean=0,
                std=0.02,
                a=-2,
                b=2,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
class DINO(SSLBase):
    def __init__(
        self, 
        backbone: torch.nn.Module, 
        max_epochs: int,
        in_dim: int,
        out_dim: int = 65536,
        momentum_teacher: float = 0.996,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 30,
        freeze_last_layer: int = 1,
        norm_last_layer: bool = True,
        use_bn_in_head: bool = False,
        hidden_dim:int = 2048, 
        bottleneck_dim:int = 256,
        concat_forward_pass: bool = False,
        **kwargs
        
    ) -> None:
        super().__init__()
        
        self.student_backbone = backbone
        self.student_head = DINOProjectionHeadV2(
            input_dim=in_dim,
            output_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            batch_norm=use_bn_in_head,
            freeze_last_layer=freeze_last_layer,
            norm_last_layer=norm_last_layer,
        )
        
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        # Remove drop path from teacher backbone
        # https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py#L166
        replace_drop_path(self.teacher_backbone, 0.0) 
        self.teacher_head = DINOProjectionHeadV2(
            input_dim=in_dim,
            output_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            batch_norm=use_bn_in_head,
        )
        self.teacher_head.load_state_dict(self.student_head.state_dict())
                
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
            
        self.loss_func = DINOLoss(
            output_dim=out_dim,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            student_temp=0.1,
            center_momentum=0.9,
        )
            

        
        self.freeze_last_layer = freeze_last_layer
        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.momentum_teacher = momentum_teacher
        self.concat_forward_pass = concat_forward_pass
        
        self.hooks = {
            Event.BATCH_END: [self.update_teacher, self.update_step],
            Event.AFTER_BACKWARD: [self.cancel_last_layer_gradients],
            Event.EPOCH_END: [self.update_epoch],
            
        }
        
    def get_momentum(self):
        return cosine_schedule(
            step=self.current_epoch, 
            max_steps=self.max_epochs, 
            start_value=self.momentum_teacher,
            end_value=1.0
            )
        
        
    def forward_concat_views(self, module: torch.nn.Module, views: torch.Tensor) -> torch.Tensor:
        """
        Adapted from https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L594
        """
        if not isinstance(views, list):
            views = [views]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in views]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(views[0].device)
        for end_idx in idx_crops:
            _out = module(torch.cat(views[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        return output.flatten(start_dim=1)
        
    def forward_teacher(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        views = inputs[:2] # Only use global views
        if self.concat_forward_pass:
            teacher_y = self.forward_concat_views(self.teacher_backbone, views)
        else:
            teacher_y = [self.teacher_backbone(view).flatten(start_dim=1) for view in views]
            teacher_y = torch.cat(teacher_y, dim=0)
        teacher_z = self.teacher_head(teacher_y)
        return torch.chunk(teacher_z, 2, dim=0)
    
    @torch.no_grad()
    def extract_features_teacher(self, input: torch.Tensor) -> Tuple[torch.Tensor]:
        teacher_y = self.teacher_backbone(input).flatten(start_dim=1)
        return teacher_y
    
    @torch.no_grad()
    def extract_features_student(self, input: torch.Tensor) -> Tuple[torch.Tensor]:
        student_y = self.teacher_backbone(input).flatten(start_dim=1)
        return student_y
    
    @torch.no_grad()
    def extract_features(self, input: torch.Tensor, backbone:Literal['student', 'teacher']='teacher') -> Tuple[torch.Tensor]:
        if backbone == 'student':
            y = self.student_backbone(input).flatten(start_dim=1)
        elif backbone == 'teacher':
            y = self.teacher_backbone(input).flatten(start_dim=1)
        else:
            raise ValueError(f"Unknown backbone {backbone}")
        return y
    
    def forward_student(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        views = inputs
        if self.concat_forward_pass:
            student_y = self.forward_concat_views(self.student_backbone, views)
        else:
            student_y = [self.student_backbone(view).flatten(start_dim=1) for view in views]
            student_y = torch.cat(student_y, dim=0)
        student_z = self.student_head(student_y)

        return torch.chunk(student_z, len(inputs), dim=0)
        
    def _forward(self, batch: Tuple[torch.Tensor, Any]) -> Tuple[torch.Tensor]:
        inputs, _ = batch
        student_z = self.forward_student(inputs)
        teacher_z = self.forward_teacher(inputs)
        
        return (student_z, teacher_z)
        
    def loss(self, outputs: torch.Tensor, batch: Tuple[Any, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        student_z, teacher_z = outputs
        if isinstance(student_z, torch.Tensor) and isinstance(self.loss_func, DINOLoss):
            student_z = student_z.chunk(10)
            teacher_z = teacher_z.chunk(2)
        if isinstance(self.loss_func, DINOLoss):
            loss_value = self.loss_func(teacher_out=teacher_z, student_out=student_z, epoch=self.current_epoch)
        else:
            loss_value = self.loss_func(teacher_output=teacher_z, student_output=student_z, epoch=self.current_epoch)
            
            
        return loss_value
        
    def update_teacher(self):
        update_momentum(self.student_backbone, self.teacher_backbone, self.get_momentum())
        update_momentum(self.student_head, self.teacher_head, self.get_momentum())
        
    def cancel_last_layer_gradients(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)
        
    def get_parameters(self, *args, **kwargs):
        # Copied from https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L632
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return {'parameters':[{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]}
    
    
class SimCLR(SSLBase):
    def __init__(
        self, 
        backbone: torch.nn.Module, 
        in_dim: int,
        out_dim: int = 128,
        hidden_dim:int = 2048, 
        num_layers: int = 2,
        use_bn_in_head: bool = True,
        concat_forward_pass: bool = False,
        temperature: float = 0.5,
        gather_distributed: bool = True,
        **kwargs
        
    ) -> None:
        super().__init__()
        
        self.backbone = backbone
        self.head = SimCLRProjectionHead(
            input_dim=in_dim,
            output_dim=out_dim,
            hidden_dim=hidden_dim,
            batch_norm=use_bn_in_head,
            num_layers=num_layers,
        )
        
        self.loss_func = NTXentLoss(
            temperature=temperature,
            gather_distributed=gather_distributed,
        )
        
        self.concat_forward_pass = concat_forward_pass
        self.current_epoch = 0
        
    
        
    def _forward(self, batch: Tuple[torch.Tensor, Any]) -> Tuple[torch.Tensor]:
        inputs, _ = batch
        
        x = inputs
        if self.concat_forward_pass:
            x = torch.cat(x, dim=0)
            h = self.backbone(x).flatten(start_dim=1)
            z = self.head(h)
            z = z.chunk(len(inputs), dim=0)
        else:
            h = [self.backbone(view).flatten(start_dim=1) for view in inputs]
            z = [self.head(view) for view in h]
        
        return tuple(z)
    
    @torch.no_grad()
    def extract_features(self, input: torch.Tensor) -> torch.Tensor:
        return self.backbone(input).flatten(start_dim=1)
        
    def loss(self, outputs: torch.Tensor, batch: Tuple[Any, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        return self.loss_func(*outputs)
        
        
    
class SSLModelHooks(Algorithm):
    def __init__(
        self,
        hooks: Dict[Event, List[callable]],
    ):
        self.hooks = hooks
        self.events = list(hooks.keys())
        
    def match(self, event: Event, state: State) -> bool:
        return event in self.events 

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event in self.events:
            for hook in self.hooks[event]:
                hook()
                
def wrap_ssl_model(model: ComposerModel, **kwargs) -> SSLBase:
    
    backbone = model.module
    
    # Reconstruct model if needed
    should_reconstruct_model = kwargs.get('should_reconstruct_model', False)
    if should_reconstruct_model:
        backbone = nn.Sequential(*list(backbone.children())[:-1])
    else:
        if hasattr(backbone, 'fc'):
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier'):
            backbone.classifier = nn.Identity()
        elif hasattr(backbone, 'head'):
            backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unknown model type {type(model)} for SSL training.")
        
    ssl_in_dim = kwargs.get('ssl_in_dim')
    if ssl_in_dim == 0:
        print("Inferring SSL input dimension from backbone")
        ssl_in_dim = backbone(torch.randn(2, 3, 224, 224)).shape[1]
        print(f"Using SSL input dimension {ssl_in_dim}")
    kwargs['in_dim'] = ssl_in_dim
    
    ssl_method = kwargs.get('ssl_method')
    loss_fn = kwargs.get('loss_fn')
    if ssl_method == 'dino':
        assert loss_fn == 'dino_loss', f"Loss must be 'dino_loss' for DINO, but got {loss_fn}"
        kwargs['max_epochs'] = int(kwargs.get('max_duration').replace('ep', ''))
        return DINO(
            backbone=backbone,
            **kwargs
        )
    elif ssl_method== 'simclr':
        assert loss_fn == 'ntxent_loss', f"Loss must be 'ntxent_loss' for SimCLR, but got {loss_fn}"
        return SimCLR(
            backbone=backbone,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown SSL method {ssl_method}")
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from torchmetrics import Metric, MetricCollection
from composer.models import ComposerClassifier
from composer.metrics import CrossEntropy, LossMetric
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class ClassifierBase(ComposerClassifier):
    def __init__(
        self,
        module,
        num_classes=1000,
        train_metrics=None,
        val_metrics=None,
        loss_fn=None,
        **kwargs
    ):
        if not val_metrics:
            val_metrics = {}
            val_metrics["MulticlassAccuracy"] = torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes, average="micro"
            )
            if isinstance(loss_fn, nn.CrossEntropyLoss) or loss_fn is None:
                val_metrics["CrossEntropyLoss"] = CrossEntropy()
            elif isinstance(loss_fn, LabelSmoothingCrossEntropy):
                loss_name = loss_fn.__class__.__name__
                val_metrics[f"{loss_name}Loss"] = LossMetric(loss_fn)
            elif isinstance(loss_fn, SoftTargetCrossEntropy):
                loss_name = loss_fn.__class__.__name__
                loss_func = lambda x, y: loss_fn(x, F.one_hot(y.long(), num_classes=num_classes))
                val_metrics[f"{loss_name}Loss"] = LossMetric(loss_func)
            elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
                loss_name = loss_fn.__class__.__name__
                loss_func = lambda x, y: loss_fn(x, F.one_hot(y.long(), num_classes=num_classes).float())
                val_metrics[f"{loss_name}"] = LossMetric(loss_func)
        val_metrics = MetricCollection(val_metrics)
        if not loss_fn:
            loss_fn = nn.CrossEntropyLoss()

        super().__init__(module, num_classes, train_metrics, val_metrics, loss_fn)

    def get_metrics(self, is_train=False):
        return {} if is_train else self.val_metrics
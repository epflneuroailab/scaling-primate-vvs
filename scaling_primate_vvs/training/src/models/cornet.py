from collections import OrderedDict

import torch
import torch.nn as nn

from .base import ClassifierBase

class CORblock_S(nn.Module):
    """
    A block of CORnet model with bottleneck structure.
    """

    def __init__(self, in_channels, out_channels, times=1, scale=4):
        super().__init__()
        self.times = times
        self.scale = scale

        # Define the layers
        self.conv_input = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.skip = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=2, bias=False
        )
        self.norm_skip = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            out_channels, out_channels * self.scale, kernel_size=1, bias=False
        )
        self.nonlin1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels * self.scale,
            out_channels * self.scale,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.nonlin2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            out_channels * self.scale, out_channels, kernel_size=1, bias=False
        )
        self.nonlin3 = nn.ReLU(inplace=True)
        self.output = nn.Identity()

        # Need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f"norm1_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm2_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm3_{t}", nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)
        for t in range(self.times):
            skip = self.norm_skip(self.skip(x)) if t == 0 else x
            self.conv2.stride = (2, 2) if t == 0 else (1, 1)

            # Sequential operations
            x = self.conv1(x)
            x = getattr(self, f"norm1_{t}")(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f"norm2_{t}")(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f"norm3_{t}")(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)
        return output


class CORnet_S_TORCH(nn.Module):
    """
    CORnet-S model.
    """

    def __init__(self, pretrained=False):
        super().__init__()

        # Define the model structure
        self.model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "V1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "conv1",
                                        nn.Conv2d(
                                            3,
                                            64,
                                            kernel_size=7,
                                            stride=2,
                                            padding=3,
                                            bias=False,
                                        ),
                                    ),
                                    ("norm1", nn.BatchNorm2d(64)),
                                    ("nonlin1", nn.ReLU(inplace=True)),
                                    (
                                        "pool",
                                        nn.MaxPool2d(
                                            kernel_size=3, stride=2, padding=1
                                        ),
                                    ),
                                    (
                                        "conv2",
                                        nn.Conv2d(
                                            64,
                                            64,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False,
                                        ),
                                    ),
                                    ("norm2", nn.BatchNorm2d(64)),
                                    ("nonlin2", nn.ReLU(inplace=True)),
                                    ("output", nn.Identity()),
                                ]
                            )
                        ),
                    ),
                    ("V2", CORblock_S(64, 128, times=2)),
                    ("V4", CORblock_S(128, 256, times=4)),
                    ("IT", CORblock_S(256, 512, times=2)),
                    (
                        "decoder",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("avgpool", nn.AdaptiveAvgPool2d(1)),
                                    ("flatten", nn.Flatten()),
                                    ("linear", nn.Linear(512, 1000)),
                                    ("output", nn.Identity()),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

        if pretrained:
            self._load_pretrained_weights()
        else:
            self._initialize_weights()

    # load pretrained weights
    def _load_pretrained_weights(self):
        # url = 'https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth'
        url = "https://cornet-models.s3.amazonaws.com/cornet_s_epoch43.pth.tar"
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location="cpu")
        # Remove 'module.' from key names (not using DataParallel)
        state_dict = OrderedDict(
            [(k.replace("module.", ""), v) for k, v in ckpt_data["state_dict"].items()]
        )
        self.model.load_state_dict(state_dict)

    def _initialize_weights(self):
        # Weight initialization
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(torch.tensor(2.0 / n)))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CORnet_S(ClassifierBase):
    def __init__(
        self,
        pretrained=False,
        num_classes=1000,
        train_metrics=None,
        val_metrics=None,
        loss_fn=None,
        **kwargs
    ):
        
        module = CORnet_S_TORCH(pretrained=pretrained).model
            
        if num_classes != 1000:
            module.decoder.linear = nn.Linear(
                in_features=module.decoder.linear.in_features,
                out_features=num_classes,
                bias=module.decoder.linear.bias is not None,
            )

        super().__init__(module, num_classes, train_metrics, val_metrics, loss_fn)
        

def create_cornet_model(**kwargs):
    arch = kwargs.get("arch", None)
    assert arch in ["cornet_s"], f"Invalid CORnet model: {arch}"
    model = CORnet_S(**kwargs)
    return model

CORNET_MODELS = [
    "cornet_s"
]
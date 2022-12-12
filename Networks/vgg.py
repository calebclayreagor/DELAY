import torch
import torch.nn as nn
from typing import List
from typing import Union
from typing import TypeVar

Self = TypeVar('Self', bound = 'VGG')

class VGG(nn.Module):
    """VGG-like network without a linear layer"""

    def __init__(self: Self, 
                 cfg: List[Union[int, str]],
                 in_channels: int
                 ) -> Self:
        super(VGG, self).__init__()
        self.features = self.make_layers(cfg, in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cfg[-1], 1)
        self._initialize_weights()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)

    def _initialize_weights(self: Self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def make_layers(self: Self,
                    cfg: List[Union[int, str]],
                    in_channels: int, 
                    negative_slope: float = 0.2
                    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size = 2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1))
                layers.append(nn.LeakyReLU(negative_slope = negative_slope))
                in_channels = v
        return nn.Sequential(*layers)
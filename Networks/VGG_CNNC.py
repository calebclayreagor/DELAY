import torch
import torch.nn as nn
from typing import List
from typing import Union
from typing import TypeVar

Self = TypeVar('Self', bound = 'VGG_CNNC')

class VGG_CNNC(nn.Module):
    """VGG-like network including a linear layer"""

    def __init__(self: Self,
                 cfg: List[Union[int, str]],
                 in_channels: int = 1
                 ) -> Self:
        super(VGG_CNNC, self).__init__()
        self.features = self.make_layers(cfg, in_channels)
        self.classifier = nn.Sequential(
                            nn.Linear(128 * 4 * 4, 512),
                            nn.ReLU(),
                            nn.Linear(512, 1))
        self._initialize_weights()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 1:
            x = torch.unsqueeze(x[:,0,:,:], 1)
        out = self.features(x)
        out = torch.flatten(out, 1)
        return self.classifier(out)

    def _initialize_weights(self: Self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def make_layers(self: Self, 
                    cfg: List[Union[int, str]], 
                    in_channels: int
                    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size = 2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1))
                layers.append(nn.ReLU())
                in_channels = v
        return nn.Sequential(*layers)
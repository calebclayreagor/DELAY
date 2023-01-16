import torch
import torch.nn as nn
from typing import List
from typing import Union
from typing import TypeVar

Self = TypeVar('Self', bound = 'SiameseVGG')

class SiameseVGG(nn.Module):
    """Siamese VGG network for primary and neighbor-gene matrices"""

    def __init__(self: Self, 
                 cfg: List[Union[int, str]], 
                 neighbors: int,
                 max_lag: int,
                 ) -> Self:
        super(SiameseVGG, self).__init__()
        self.in_channels = (1 + max_lag)
        self.primary_features = self.make_layers(cfg, self.in_channels)
        self.neighbor_features = self.make_layers(cfg, self.in_channels)
        self.primary_embedding = nn.Sequential(
                                    nn.Linear(128 * 4 * 4, 512),
                                    nn.ReLU())
        self.neighbor_embedding = nn.Sequential(
                                    nn.Linear(128 * 4 * 4, 512),
                                    nn.ReLU())              
        self.classifier = nn.Sequential(
                            nn.Linear(512 * (3+2*neighbors), 512),
                            nn.ReLU(),
                            nn.Linear(512, 128),
                            nn.ReLU(),
                            nn.Linear(128, 1))
        self._initialize_weights()

    def forward_primary(self: Self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 4: x = torch.unsqueeze(x, 1)
        out = self.primary_features(x)
        out = torch.flatten(out, 1)
        return self.primary_embedding(out)

    def forward_neighbor(self: Self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 4: x = torch.unsqueeze(x, 1)
        out = self.neighbor_features(x)
        out = torch.flatten(out, 1)
        return self.neighbor_embedding(out)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        primary = x[:, :self.in_channels, :, :]
        neighbors = x[:, (self.in_channels + 1):, :, :]
        out = self.forward_primary(primary)
        for idx in range(0, neighbors.size(1), self.in_channels):
            out2 = neighbors[:, idx : (idx + self.in_channels + 1), :, :]
            out2 = self.forward_neighbor(out2)
            out = torch.cat([out, out2], axis=1)
        return self.classifier(out)

    def _initialize_weights(self: Self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def make_layers(self: Self,
                    cfg: List[Union[int, str]], 
                    in_channels: int,
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
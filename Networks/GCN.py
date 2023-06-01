import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Sequential
from typing import List
from typing import TypeVar

Self = TypeVar('Self', bound = 'GCN')

class GCN(nn.Module):
    """GCN classifier"""

    def __init__(self: Self,
                 cfg: List[int],
                 in_channels: int,
                 ) -> Self:
        super(GCN, self).__init__()
        self.features = self.make_layers(cfg, in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cfg[-1], 1)
        self._initialize_weights()

    def forward(self: Self,
                x: torch.Tensor,
                # edge_index: torch.Tensor
                ) -> torch.Tensor:
        out = torch.zeros(x.size(0), 1, device = torch.cuda.current_device())
        edge_index = torch.tensor([[0,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5, 
                                    5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  8,  8,  8,  9,  9,  9,
                                    10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15,
                                    15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 20, 21, 21,
                                    21, 22, 23, 23, 23, 24, 25, 26],
                                   [1,  0,  2,  4,  6,  1,  3,  7,  2,  4, 24,  1,  3,  5,  4,  6,
                                    17, 19,  1,  5,  7,  8, 10,  2,  6, 23,  6,  9, 16,  8, 10, 13,
                                    6,  9, 11, 10, 12, 23, 11, 13, 21,  9, 12, 14, 13, 15, 20, 14,
                                    16, 18,  8, 15, 17,  5, 16, 18, 15, 17, 25, 26,  5, 14, 12, 22,
                                    23, 21, 7, 11, 21, 3, 18, 18]], 
                                   dtype = torch.long, device = torch.cuda.current_device())
        for i in range(x.size(0)):
            xi = x[i, ...]
            xi = torch.tile(xi, (27, 1, 1, 1))
            xi = self.features(xi, edge_index)
            xi = self.avgpool(xi)
            xi = torch.flatten(xi, 1)
            out[i] = self.classifier(xi)[0]
        return out
    
    def _initialize_weights(self: Self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def make_layers(self: Self, 
                    cfg: List[int], 
                    in_channels: int,
                    negative_slope: float = 0.2
                    ) -> Sequential:
        layers: List[nn.Module] = []
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size = 2))
            else:
                layers.append((Conv2dMessage(in_channels, v), 'x, edge_index -> x'))
                layers.append(nn.LeakyReLU(negative_slope = negative_slope, inplace = True))
                in_channels = v
        return Sequential('x, edge_index', layers)

class Conv2dMessage(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = 'add', node_dim = 0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        # self.bias = nn.Parameter(torch.Tensor(out_channels))
        self._initialize_weights()

    def _initialize_weights(self: Self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Parameter):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x, edge_index):
        out = self.conv(x)
        out = self.propagate(edge_index, x = out)
        # out += self.bias[None, :, None, None]
        return out
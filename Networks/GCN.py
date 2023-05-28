import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential
from typing import List
from typing import TypeVar

Self = TypeVar('Self', bound = 'GCN')

class GCN(nn.Module):
    """GCN classifier"""

    def __init__(self: Self,
                 cfg: List[int],
                 in_dimension: int,
                 ) -> Self:
        super(GCN, self).__init__()
        self.features = self.make_layers(cfg, in_dimension)
        self.classifier = nn.Linear(cfg[-1], 1)
        # self._initialize_weights()

    def forward(self: Self,
                x: torch.Tensor,
                # edge_index: torch.Tensor
                ) -> torch.Tensor:
        out = torch.zeros(x.size(0), device = torch.cuda.current_device())
        x = torch.flatten(x, start_dim = 1)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
                                   [1, 2, 0, 2, 0, 1]], dtype = torch.long,
                                   device = torch.cuda.current_device())    ##
        for i in range(x.size(0)):
            xi = torch.tile(x[i, :], (3, 1))   ##
            xi = self.features(xi, edge_index)
            out[i] = self.classifier(xi)[0]
        return out

    # def _initialize_weights(self: Self) -> None:
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_uniform_(m.weight)

    def make_layers(self: Self, 
                    cfg: List[int], 
                    in_dimension: int
                    ) -> Sequential:
        layers: List[nn.Module] = []
        for v in cfg:
            layers.append((GCNConv(in_dimension, v, add_self_loops = False, normalize = False), 'x, edge_index -> x'))
            layers.append(nn.ReLU(inplace = True))
            in_dimension = v
        return Sequential('x, edge_index', layers)
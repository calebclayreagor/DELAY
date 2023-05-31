import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential
from typing import List
from typing import TypeVar
import numpy as np

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
                                    23, 21, 7, 11, 21,  3, 18, 18]], 
                                   dtype = torch.long, device = torch.cuda.current_device())
        ## CURRENTLY, USING EDGES INSTEAD OF NODES
        # loop over examples
        n = (edge_index.size(1) - 1)
        for i in range(x.size(0)):
            try:
                ii = np.random.choice(x.size(-1), n * 10, replace = False)
            except:
                ii = np.random.choice(x.size(-1), n * 10, replace = True)
            xi = x[i, ..., ii]
            # loop over single cells
            for j in range(0, xi.size(-1), n):
                xij1 = torch.squeeze(xi[..., j:(j + n)]).T
                if j == 0:
                    xij0 = torch.zeros(1, xi.size(0), device = torch.cuda.current_device())
                else:
                    xij0 = xij[0].reshape(1, -1)
                    # xij1 = xij[1:, :] + xij1
                xij = torch.concat((xij0, xij1), dim = 0)
                xij = self.features(xij, edge_index)
            out[i] = self.classifier(xij)[0]
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
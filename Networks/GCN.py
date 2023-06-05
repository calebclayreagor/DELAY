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
                 in_dimensions: int,
                 ) -> Self:
        super(GCN, self).__init__()
        self.features = self.make_layers(cfg, in_dimensions)
        self.classifier = nn.Linear(cfg[-1], 1)                     
        # self._initialize_weights()

    def forward(self: Self,
                x: torch.Tensor,
                # edge_index: torch.Tensor
                ) -> torch.Tensor:
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
            xi = x[i, ...]                      # [nchan, nbins, nbins]
            # id = np.indices(xi.size())          # [3, nchan, nbins, nbins]
            # id = id.astype(np.float64)
            # id[0, ...] /= id.shape[1]
            # id[1:, ...] /= id.shape[-1]
            # id = torch.tensor(id, dtype = torch.long, device = torch.cuda.current_device())
            # xi = torch.unsqueeze(xi, 0)         # [1, nchan, nbins, nbins]
            # xi = torch.cat((xi, id), dim = 0)   # [4, nchan, nbins, nbins]
            xi = torch.flatten(xi)              # [4 * nchan * nbins * nbins]
            xi = torch.unsqueeze(xi, 0)         # [1, 4 * nchan * nbins * nbins]
            xi = torch.tile(xi, (27, 1))        # [27, 4 * nchan * nbins * nbins]
            if i == 0:
                x_batch = xi
                edge_index_batch = edge_index
            else:
                x_batch = torch.cat((x_batch, xi), dim = 0)
                edge_index_batch = torch.cat((edge_index_batch, (27 * i) + edge_index), dim = 1)
        out = self.features(x_batch, edge_index_batch)
        out = out[::27, ...]                    # [batch_size, cfg[-1]]
        return self.classifier(out)
    
    # def _initialize_weights(self: Self) -> None:
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_uniform_(m.weight)

    def make_layers(self: Self, 
                    cfg: List[int], 
                    in_dimensions: int,
                    negative_slope: float = 0.2
                    ) -> Sequential:
        layers: List[nn.Module] = []
        for v in cfg:
            layers.append((GCNConv(in_dimensions, v, add_self_loops = False, normalize = False), 'x, edge_index -> x'))
            layers.append(nn.LeakyReLU(negative_slope = negative_slope, inplace = True))
            in_dimensions = v
        return Sequential('x, edge_index', layers)
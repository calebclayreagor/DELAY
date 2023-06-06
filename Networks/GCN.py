import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential
from typing import List
from typing import TypeVar
import numpy as np
import networkx as nx

Self = TypeVar('Self', bound = 'GCN')

class GCN(nn.Module):
    """GCN classifier"""

    def __init__(self: Self,
                 graph: np.array,
                 cfg: List[int],
                 in_dimensions: int,
                 ) -> Self:
        super(GCN, self).__init__()
        G = nx.from_numpy_array(graph, create_using = nx.DiGraph)
        n_conv = 0
        for node in list(G.nodes()):
            d = nx.shortest_path_length(G, node, target = 0, weight = None)
            if d > n_conv: n_conv = d
        cfg = [cfg[0]] * n_conv
        edge_index = np.array(np.where(graph > 0))
        edge_weight = graph[edge_index[0], edge_index[1]]
        self.edge_index = torch.tensor(edge_index, dtype = torch.long)
        self.edge_weight = torch.tensor(edge_weight, dtype = torch.double)
        self.features = self.make_layers(cfg, in_dimensions)
        self.classifier = nn.Linear(cfg[-1], 1)                 
        self._initialize_weights()

    def _initialize_weights(self: Self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        n_nodes = (self.edge_index.max() + 1)
        for i in range(x.size(0)):
            xi = x[i, ...]                          # [nchan, nbins, nbins]     (torch.float32)
            id = np.indices(xi.size())              # [3, nchan, nbins, nbins]
            id = id.astype(np.float32)
            id[0, ...] /= id.shape[1]
            id[1:, ...] /= id.shape[-1]
            id = torch.tensor(id, dtype = torch.float, device = torch.cuda.current_device())
            xi = torch.unsqueeze(xi, 0)             # [1, nchan, nbins, nbins]
            xi = torch.cat((xi, id), dim = 0)       # [4, nchan, nbins, nbins]
            xi = torch.flatten(xi)                  # [4 * nchan * nbins * nbins]
            xi = torch.unsqueeze(xi, 0)             # [1, 4 * nchan * nbins * nbins]
            xi0 = torch.zeros(xi.size(), dtype = torch.float, device = torch.cuda.current_device())
            xi = torch.tile(xi, (n_nodes - 1, 1))   # [n_nodes - 1, 4 * nchan * nbins * nbins]
            xi = torch.cat((xi0, xi), dim = 0)      # [n_nodes, 4 * nchan * nbins * nbins]
            if i == 0:
                x_batch = xi
                edge_index_batch = self.edge_index
                edge_weight_batch = self.edge_weight
            else:
                x_batch = torch.cat((x_batch, xi), dim = 0)
                edge_index_batch = torch.cat((edge_index_batch, (n_nodes * i) + self.edge_index), dim = 1)
                edge_weight_batch = torch.cat((edge_weight_batch, self.edge_weight), dim = 0)

        print(x_batch.size())
        input(x_batch)

        print(edge_index_batch.size())
        input(edge_index_batch)

        print(edge_weight_batch.size())
        input(edge_weight_batch)

        out = self.features(x_batch, edge_index_batch, edge_weight_batch)
        out = out[::n_nodes, ...]                   # [batch_size, cfg[-1]]
        return self.classifier(out)

    def make_layers(self: Self,
                    cfg: List[int],
                    in_dimensions: int,
                    negative_slope: float = 0.2
                    ) -> Sequential:
        layers: List[nn.Module] = []
        for v in cfg:
            layers.append((GCNConv(in_dimensions, v, add_self_loops = False, normalize = False), 'x, edge_index, edge_weight -> x'))
            layers.append(nn.LeakyReLU(negative_slope = negative_slope, inplace = True))
            in_dimensions = v
        return Sequential('x, edge_index, edge_weight', layers)
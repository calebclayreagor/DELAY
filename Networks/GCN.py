import torch
import torch.nn as nn
import pathlib, glob
import numpy as np
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential
from typing import List
from typing import TypeVar

Self = TypeVar('Self', bound = 'GCN')

class GCN(nn.Module):
    """GCN classifier"""

    def __init__(self: Self,
                 graphs: str,
                 cfg: List[int],
                 in_dimensions: int,
                 ) -> Self:
        super(GCN, self).__init__()

        graphs = sorted(list(map(str, pathlib.Path(graphs).glob('*.csv'))))
        self.graphs = [np.loadtxt(graph, delimiter = ',', dtype = np.int64) for graph in graphs]
        print(self.graphs)
        cfg = cfg[0]; self.n_conv = 0
        for graph in self.graphs:
            G = nx.MultiDiGraph()
            G.add_edges_from(graph.T)
            for node in list(G.nodes()):
                d = nx.shortest_path_length(G, node, 0, weight = None)
                if d > self.n_conv: self.n_conv = d
        input(self.n_conv)
        
        self.edge_index = torch.tensor(np.array(np.where(graph)), dtype = torch.long)
        self.n_nodes = (self.edge_index.max() + 1)
        self.embedding = nn.Sequential(nn.Linear(in_dimensions, cfg), nn.ReLU(inplace = True))
        self.features = Sequential('x, edge_index',
            [(GCNConv(cfg, cfg, add_self_loops = False, normalize = False), 'x, edge_index -> x'),
             nn.ReLU(inplace = True)])
        self.classifier = nn.Linear(cfg, 1)
        self._initialize_weights()

    def _initialize_weights(self: Self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        edge_index = self.edge_index.to(torch.cuda.current_device())
        for i in range(x.size(0)):
            xi = x[i, ...]                           # [nchan, nbins, nbins]      (torch.float32)
            xi = torch.flatten(xi)                   # [nchan * nbins * nbins]
            xi = torch.unsqueeze(xi, 0)              # [1, nchan * nbins * nbins]
            xi = torch.tile(xi, (self.n_nodes, 1))   # [n_nodes, nchan * nbins * nbins]
            if i == 0:
                x_batch = xi
                edge_index_batch = edge_index
            else:
                x_batch = torch.cat((x_batch, xi), dim = 0)
                edge_index_batch = torch.cat(
                    (edge_index_batch, (self.n_nodes * i) + edge_index), dim = 1)
        out = self.embedding(x_batch)
        for _ in range(self.n_conv):
            out = self.features(out, edge_index_batch)
        out = out[::self.n_nodes, ...]
        return self.classifier(out)
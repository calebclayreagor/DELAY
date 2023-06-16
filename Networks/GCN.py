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

        # compile list of edge arrays
        graphs = sorted(list(map(str, pathlib.Path(graphs).glob('*.csv'))))
        graphs = [np.loadtxt(graph, delimiter = ',', dtype = np.int64) for graph in graphs]
        self.n_nodes = np.array([(graph.max() + 1) for graph in graphs])

        # find max required n_convs
        cfg = 1
        self.n_conv = 0
        for graph in graphs:
            G = nx.MultiDiGraph()
            G.add_edges_from(graph.T)
            for node in list(G.nodes()):
                d = nx.shortest_path_length(G, node, 0, weight = None)
                if d > self.n_conv: self.n_conv = d
        
        # compile master edge array
        for i in range(len(graphs)):
            graph_i = torch.tensor(graphs[i], dtype = torch.long)
            if i == 0: self.edge_index = graph_i
            else:
                graph_i += self.n_nodes[:i].sum()
                self.edge_index = torch.cat((self.edge_index, graph_i), dim = 1)

        # neural network architecture
        self.embedding = nn.Sequential(nn.Linear(in_dimensions, cfg), nn.ReLU(inplace = True))
        self.features = Sequential('x, edge_index',
            [(GCNConv(cfg, cfg, add_self_loops = False, normalize = False), 'x, edge_index -> x'),
             nn.ReLU(inplace = True)])
        # self.classifier = nn.Linear(cfg, 1)
        self._initialize_weights()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        edge_index = self.edge_index.to(torch.cuda.current_device())
        target_ix = np.concatenate((np.array([0]), (np.cumsum(self.n_nodes)[:-1])))
        target_ix = torch.tensor(target_ix, dtype = torch.long)
        target_ix = target_ix.to(device = torch.cuda.current_device())
        # loop over pseudotime
        for t in range(x.size(-1)):
            # loop over mini-batch
            for i in range(x.size(0)):
                xi = torch.unsqueeze(x[i, :, 0, t] , 0)                            # [1, n_genes]
                xi = torch.tile(xi, (self.n_nodes.sum(), 1))                       # [n_nodes, n_genes]
                if i == 0:
                    x_batch = xi
                    if t == 0:
                        edge_index_batch = edge_index
                        target_ix_batch = target_ix
                else:
                    x_batch = torch.cat((x_batch, xi), dim = 0)
                    if t == 0:
                        edge_index_batch = torch.cat(
                            (edge_index_batch, (self.n_nodes.sum() * i) + edge_index), dim = 1)
                        target_ix_batch = torch.cat(
                            (target_ix_batch, (self.n_nodes.sum() * i) + target_ix), dim = 0)    
            if t == 0:
                x_batch = self.embedding(x_batch)
            else:
                embed_ix = torch.isin(
                    torch.arange(x_batch.size(0), device = torch.cuda.current_device()), 
                    target_ix_batch)
                x_batch_t = torch.zeros(x_batch.size(0), 1, dtype = x_batch.dtype,
                                        device = torch.cuda.current_device())
                x_batch_t[embed_ix] = self.embedding(x_batch[embed_ix, :])
                x_batch_t[target_ix_batch] = out
                x_batch = x_batch_t
            for _ in range(self.n_conv): 
                x_batch = self.features(x_batch, edge_index_batch)
            out = torch.split(x_batch, [self.n_nodes.sum()] * x.size(0))           # len(batch_size): [n_nodes, n_genes]
            out = torch.concat([out_i[target_ix, :] for out_i in out], dim = 0)    # [n_graphs * batch_size, n_genes]
        # out = self.classifier(out)                                                 # [n_graphs * batch_size, 1]
        out = torch.split(out, [len(self.n_nodes)] * x.size(0))                    # len(batch_size): [n_graphs, 1]
        out = torch.concat(out, dim = 1).mean(axis = 0).reshape(-1, 1)             # [batch_size, 1]
        return out
    
    def _initialize_weights(self: Self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
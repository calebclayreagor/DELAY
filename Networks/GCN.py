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
                 in_channels: int,
                 nbins: int,
                 ) -> Self:
        super(GCN, self).__init__()

        # compile list of edge arrays
        graphs = sorted(list(map(str, pathlib.Path(graphs).glob('*.csv'))))
        graphs = [np.loadtxt(graph, delimiter = ',', dtype = np.int64) for graph in graphs]
        self.n_nodes = np.array([(graph.max() + 1) for graph in graphs])

        # find max required n_convs
        cfg = cfg[0]; self.n_conv = 0
        for graph in graphs:
            G = nx.MultiDiGraph()
            G.add_edges_from(graph.T)
            for node in list(G.nodes()):
                d = nx.shortest_path_length(G, node, 0, weight = None)
                if d > self.n_conv: self.n_conv = d
        
        # compile edge_index, x_ind: graphs x channels
        for i in range(len(graphs)):
            graph_i = torch.tensor(graphs[i], dtype = torch.long)
            x_ind = torch.ones(self.n_nodes[i], dtype = torch.long)
            for j in range(in_channels):
                if j == 0:
                    graph_i_channel = graph_i
                    x_ind_channel = (x_ind * j)
                else:
                    graph_i_channel = torch.cat(
                        (graph_i_channel, (self.n_nodes[i] * j) + graph_i), dim = 1)
                    x_ind_channel = torch.cat((x_ind_channel, (x_ind * j)), dim = 0)
            if i == 0:
                self.edge_index = graph_i_channel
                self.x_ind = x_ind_channel
            else:
                graph_i_channel += (self.n_nodes[:i].sum() * in_channels)
                self.edge_index = torch.cat((self.edge_index, graph_i_channel), dim = 1)
                self.x_ind = torch.cat((self.x_ind, x_ind_channel), dim = 0)

        # neural network architecture
        self.embedding = nn.Sequential(nn.Linear((nbins ** 2), cfg), nn.ReLU(inplace = True))
        self.features = Sequential('x, edge_index',
            [(GCNConv(cfg, cfg, add_self_loops = False, normalize = False), 'x, edge_index -> x'),
             nn.ReLU(inplace = True)])
        self.classifier = nn.Linear((in_channels * cfg), 1)
        self._initialize_weights()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        edge_index = self.edge_index.to(torch.cuda.current_device())
        x_ind = self.x_ind.to(torch.cuda.current_device())
        for i in range(x.size(0)):
            xi = x[i, ...]                                                               # [nchan, nbins, nbins]   (torch.float32)
            xi = torch.flatten(xi, 1)                                                    # [nchan, nbins * nbins]
            xi = xi[x_ind, :]                                                            # [nchan * n_nodes, nbins * nbins]
            if i == 0:
                x_batch = xi
                edge_index_batch = edge_index
            else:
                x_batch = torch.cat((x_batch, xi), dim = 0)
                edge_index_batch = torch.cat(
                    (edge_index_batch, ((self.n_nodes.sum() * x.size(1)) * i) + edge_index), dim = 1)
        out = self.embedding(x_batch)                                                    # [nchan * n_nodes * batch_size, cfg]
        for _ in range(self.n_conv):
            out = self.features(out, edge_index_batch)
        out = list(torch.split(out, [self.n_nodes.sum() * x.size(1)] * x.size(0)))       # len(batch_size)   [nchan * n_nodes, cfg]
        for i in range(len(out)):
            out[i] = list(torch.split(out[i], list(self.n_nodes * x.size(1))))           #    len(n_graphs)  [n_nodes_graph * nchan, cfg]
            for j in range(len(out[i])):
                out[i][j] = list(torch.split(out[i][j], [self.n_nodes[j]] * x.size(1)))  #       len(nchan)  [n_nodes_graph, cfg]

                print(len(out[i][j]))
                input(out[i][j][0].size())




            out[i] = list(map(lambda j: torch.flatten(j).reshape(1, -1), out[i]))        #         len(n_nodes)  [1, nchan * cfg]
            out[i] = torch.cat(out[i], dim = 0)                                          #         [n_nodes, nchan * cfg]
        out_ix = np.concatenate((np.array([0]), (np.cumsum(self.n_nodes)[:-1])))
        out = torch.concat([out_i[out_ix, :] for out_i in out], dim = 0)                 # [n_graphs * batch_size, nchan * cfg]
        out = self.classifier(out)                                                       # [n_graphs * batch_size, 1]
        out = torch.split(out, [len(self.n_nodes)] * x.size(0))                          # len(batch_size)   [n_graphs, 1]
        out = torch.concat(out, dim = 1).mean(axis = 0).reshape(-1, 1)                   # [batch_size, 1]
        return out
    
    def _initialize_weights(self: Self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
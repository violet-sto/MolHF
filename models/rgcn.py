import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import T
from models.basic import Rescale, TensorLinear

class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_edge_type=4, normalize=True):
        super(GraphConv, self).__init__()

        self.graph_linear_self = nn.Linear(in_channels, out_channels)
        self.graph_linear_edge = nn.Linear(
            in_channels, out_channels * num_edge_type)
        self.normalize = normalize
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def forward(self, h, adj):
        """
        graph convolution over batch and multi-graphs
        :return:
        """
        if self.normalize:
            normalized_adj = rescale_adj(adj)  # all is better than view
        else:
            normalized_adj = adj
        mb, node, ch = h.shape  
        hs = self.graph_linear_self(h) 
        m = self.graph_linear_edge(h)
        m = m.reshape(mb, node, self.out_ch, self.num_edge_type)
        m = m.permute(0, 3, 1, 2)  
        hr = torch.matmul(normalized_adj, m)
        hr = hr.sum(dim=1)  
        return hs + hr  


class RGCN(nn.Module):
    def __init__(self, num_layers, n_node, in_dim, hid_dim, out_dim, dropout=0., normalization=True, pool=False):
        '''
        :num_layars: the number of layers in each R-GCN
        '''
        super(RGCN, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalization = normalization
        self.relu = nn.ReLU()
        self.rescale = Rescale()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.num_layers):
            if i:
                self.convs.append(GraphConv(hid_dim, hid_dim, normalize=True))

            else:
                self.convs.append(
                    GraphConv(in_dim//2, hid_dim, normalize=True))

            if normalization:
                self.bns.append(nn.BatchNorm1d(n_node))

        self.linear1 = nn.Linear(hid_dim, hid_dim)
        self.bn1 = nn.BatchNorm1d(n_node)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.pool = pool

    def forward(self, x, adj):
        '''
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        :return:
        '''
        batch_size, n_node, _ = x.shape

        if adj[0][0][0][1] == adj[0][0][1][0]:
            adj_sym = adj

        else:
            adj_sym = (adj + adj.permute(0, 1, 3, 2))/2
            if n_node == 40 or n_node == 20:
                adj_sym = torch.floor(adj_sym)
            elif n_node == 10:
                adj_sym = torch.round(adj_sym*8)
            elif n_node == 5:
                adj_sym = torch.round(adj_sym*16)

        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, adj_sym)
            if self.normalization:
                h = self.bns[i](h)
            h = self.relu(h)

        h = self.linear1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.linear2(h)
        if self.pool:
            h = h.mean(1)
        return h

def rescale_adj(adj, num_nodes=None, improved=False, add_self_loops=False, type='all'):
    adj = adj.clone()  # !keep raw adj stable

    if num_nodes is None:
        num_nodes = adj.shape[-1]

    if add_self_loops:
        fill_value = 2. if improved else 1.
        adj += (torch.eye(num_nodes)*fill_value).to(adj.device)
    else:
        adj[:, :, range(num_nodes), range(num_nodes)] = 0

    # TODO D^{-1/2}*A*D^{-1/2}

    if type == 'view':
        out_degree = adj.sum(dim=-1)
        out_degree_sqrt_inv = out_degree.pow(-1)
        out_degree_sqrt_inv[out_degree_sqrt_inv == float('inf')] = 0
        adj_prime = out_degree_sqrt_inv.unsqueeze(-1) * adj
    else:  # default type all
        num_neighbors = adj.sum(dim=(1, 2)).float()
        num_neighbors_inv = num_neighbors.pow(-1)
        num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
        adj_prime = num_neighbors_inv[:, None, None, :] * adj

    return adj_prime

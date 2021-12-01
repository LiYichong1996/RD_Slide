import torch
import torch_scatter
from torch import nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import conv as conv_g
from torch_geometric.nn import pool as pool_g


class Backbone(nn.Module):
    def __init__(self, in_size, out_size, hide_size_list, normalize=False, bias=False):
        super(Backbone, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.size_list = [in_size] + hide_size_list + [out_size]

        self.layer_gcn = []
        for i in range(len(self.size_list) - 1):
            self.layer_gcn.append(
                conv_g.GCNConv(self.size_list[i], self.size_list[i+1], normalize=normalize, bias=bias)
            )
        self.layer_gcn = nn.ModuleList(self.layer_gcn)

    def forward(self, x, edge_index, edge_w=None):
        for layer in self.layer_gcn:
            x = layer(x, edge_index, edge_w)
            x = F.leaky_relu(x)
        return x


class Actor(nn.Module):
    def __init__(self, in_size, out_size, hide_size_list, n_gcn, normalize=False, bias=False):
        super(Actor, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.n_gcn = n_gcn
        self.size_list = [in_size] + self.hide_size_list + [out_size]
        self.normalize = normalize

        self.gcn_layers = []
        for i in range(n_gcn):
            self.gcn_layers.append(
                conv_g.GCNConv(self.size_list[i], self.size_list[i+1], bias=bias)
            )
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        self.fc1 = nn.Linear(self.size_list[self.n_gcn], self.size_list[self.n_gcn+1], bias=bias)
        # self.norm1 = nn.BatchNorm1d(self.size_list[self.n_gcn+1])
        self.fc2 = nn.Linear(self.size_list[self.n_gcn+1], self.size_list[-1], bias=bias)
        # self.norm2 = nn.BatchNorm1d(self.size_list[-1])
        if self.normalize:
            self.norm1 = nn.BatchNorm1d(self.size_list[self.n_gcn + 1])
            self.norm2 = nn.BatchNorm1d(self.size_list[-1])

    def forward(self, x, edge_index, edge_w=None):
        for layer in self.gcn_layers:
            x = layer(x, edge_index, edge_w)
            x = F.leaky_relu(x)

        if self.normalize:
            y = F.leaky_relu(self.fc1(x))
            y = self.norm1(y)
            y = F.leaky_relu(self.fc2(y))
            y = self.norm2(y)
        else:
            y = F.leaky_relu((self.fc1(x)))
            y = F.leaky_relu((self.fc2(y)))

        node_actions = F.softmax(y, dim=1)

        return node_actions


class Critic(nn.Module):
    def __init__(self, in_size, out_size, hide_size_list, n_gcn, normalize=False, bias=False):
        super(Critic, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.n_gcn = n_gcn
        self.size_list = [in_size] + self.hide_size_list + [out_size]
        self.normalize = normalize

        self.gcn_layers = []
        for i in range(n_gcn):
            self.gcn_layers.append(
                conv_g.GCNConv(self.size_list[i], self.size_list[i + 1], bias=bias)
            )
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        self.fc1 = nn.Linear(self.size_list[self.n_gcn], self.size_list[self.n_gcn + 1], bias=bias)
        # self.norm1 = nn.BatchNorm1d(self.size_list[self.n_gcn+1])
        self.fc2 = nn.Linear(self.size_list[self.n_gcn + 1], self.size_list[-1], bias=bias)
        # self.norm2 = nn.BatchNorm1d(self.size_list[-1])
        if self.normalize:
            self.norm1 = nn.BatchNorm1d(self.size_list[self.n_gcn + 1])
            self.norm2 = nn.BatchNorm1d(self.size_list[-1])

    def forward(self, x, edge_index, batch, edge_w=None):
        for layer in self.gcn_layers:
            x = layer(x, edge_index, edge_w)
            x = F.leaky_relu(x)

        if self.normalize:
            y = F.leaky_relu(self.fc1(x))
            y = self.norm1(y)
            y = F.leaky_relu(self.fc2(y))
            y = self.norm2(y)
        else:
            y = F.leaky_relu((self.fc1(x)))
            y = F.leaky_relu((self.fc2(y)))

        value = pool_g.avg_pool_x(batch, y, batch)[0]

        return value


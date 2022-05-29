"""
    Utility file to select GraphNN model as
    selected by the user
"""


from nets.superpixels_graph_classification.gcn_net import GCNNet
from nets.superpixels_graph_classification.gat_net import GATNet
from nets.superpixels_graph_classification.gin_net import GINNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from layers.gcn_layer import GCNLayer
from layers.gin_layer import GINLayer


def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GIN(net_params):
    return GINNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        
        'GCN': GCN,
        'GAT': GAT,
        'GIN': GIN
        
    }
        
    return models[MODEL_NAME](net_params)


class explainer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = 1
        hidden_dim = 16
        dropout = 0.0

        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']

        self.layers = nn.ModuleList([GCNLayer(in_dim, hidden_dim, F.relu, dropout, self.graph_norm, self.batch_norm,
                                              self.residual)])
        self.layers.append(GCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.graph_norm, self.batch_norm, self.residual))
        self.layers.append(GCNLayer(hidden_dim, 1, None, dropout, self.graph_norm, self.batch_norm, self.residual))

    def forward(self, g, h, snorm_n):

        h_ = 0
        h_ += h
        for conv in self.layers:
            h = conv(g, h, snorm_n)
            h_ += torch.unsqueeze(torch.mean(h, dim=1), 1)
        # g.ndata['h'] = h_
        # h = dgl.softmax_nodes(g, 'h')
        # node_num = dgl.broadcast_nodes(g, torch.tensor(g.batch_num_nodes))
        # node_num = torch.unsqueeze(node_num, 1).to(h.device)
        # h = torch.mul(h, node_num)
        return F.relu(h_)

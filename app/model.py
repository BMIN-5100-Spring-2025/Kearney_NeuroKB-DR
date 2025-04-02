"""
This file defines a link prediction model using SAGEConv and
an encoder and decoder. The model works on heterogenous graphs.
"""
###
# IMPORTS
###

from torch_geometric.nn import SAGEConv,to_hetero
import os
import torch
import torch_geometric.transforms as T
from torch.nn import Linear
import torch_geometric

project_path = os.getcwd()

###
# LOAD & SPLIT DATA
###

base_directory = os.path.dirname(os.path.dirname(__file__))
db_directory = os.getenv('DB_DIR', os.path.join(base_directory, 'data/db/'))

device = torch_geometric.device('auto')
data = torch.load(f'{db_directory}neuroKB.pth')
# print(db_directory)
# data = torch.load(f'Files/data/db/neuroKB.pth')
data = T.ToUndirected()(data).to(device)

###
# DEFINE MODEL
###

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
    def forward(self, x, edge_index):
        # TODO - could adjust relu to something else
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['Drug'][row], z_dict['Disease'][col]], dim=-1)
        z = self.lin1(z)
        z = torch.sigmoid(z)
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
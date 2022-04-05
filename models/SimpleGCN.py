import torch
from torch_geometric.nn import NNConv


class SimpleMPGNN(torch.nn.Module):
    def __init__(self, n_node_features, n_out_classes, n_edge_features):
        super().__init__()
        n_i = n_edge_features
        n_o = n_node_features * 16
        n_h = int((n_i + n_o)/2)
        mlp_1 = torch.nn.Sequential(torch.nn.Linear(n_i, n_h), torch.nn.ReLU(), torch.nn.Linear(n_h, n_o))
        self.conv1 = NNConv(n_node_features, 16, mlp_1)

        n_i = n_edge_features
        n_o = 16 * n_out_classes
        n_h = int((n_i + n_o)/2)
        mlp_2 = torch.nn.Sequential(torch.nn.Linear(n_i, n_h), torch.nn.ReLU(), torch.nn.Linear(n_h, n_o))
        self.conv2 = NNConv(16, n_out_classes, mlp_2)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout()
        self.softmax1 = torch.nn.LogSoftmax(dim=1)

    def forward(self, x_in, edge_index, edge_atts):
        x1 = self.dropout1(self.relu1(self.conv1(x_in, edge_index, edge_atts)))
        x2 = self.conv2(x1, edge_index, edge_atts)
        x_out = self.softmax1(x2)
        return x_out

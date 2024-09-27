import torch
import torch.nn as nn
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, node_features, hidden_dim, num_classes, drop_out, num_layers, use_bias):
        """
        Args:
            node_features (int): Number of features per node
            hidden_dim (int): Number of hidden units in each GCN layer
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
            num_layers (int): Number of GCN layers
            use_bias (bool): Whether to use bias in the layers
        """
        super(GCN, self).__init__()

        self.dropout = nn.Dropout(drop_out)

        # First layer 
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(node_features, hidden_dim, bias=use_bias))

        # Subsequent 
        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, bias=use_bias))

        # Last layer
        self.layers.append(GraphConv(hidden_dim, num_classes, bias=use_bias))

    def forward(self, graph, features):
        x = features
        for current, layer in enumerate(self.layers):
            x = layer(graph, x)
            if current != len(self.layers) - 1: 
                x = torch.relu(x)
                x = self.dropout(x)
        return x
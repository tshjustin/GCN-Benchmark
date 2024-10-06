import torch
import torch.nn as nn
from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, dropout, num_layers, use_bias, aggregator_type='mean'):
        super(GraphSAGE, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        # Input layer: input features -> hidden_dim
        self.layers.append(SAGEConv(in_feats, hidden_dim, aggregator_type=aggregator_type, bias=use_bias))

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type=aggregator_type, bias=use_bias))

        # Output layer: hidden_dim -> num_classes
        self.layers.append(SAGEConv(hidden_dim, num_classes, aggregator_type=aggregator_type, bias=use_bias))

    def forward(self, graph, features):
        x = features
        for i, layer in enumerate(self.layers):
            x = layer(graph, x)
            if i != len(self.layers) - 1:  # Apply ReLU and dropout for all but the last layer
                x = torch.relu(x)
                x = self.dropout(x)
        return x.squeeze()

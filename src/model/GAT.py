import torch
import torch.nn as nn
from dgl.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, num_heads, dropout, num_layers, use_bias=True):
        super(GAT, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Calculate hidden_size per head - Ensure same as GCN number of layers of 16
        hidden_size = hidden_dim // num_heads 

        # First GAT layer: input features -> hidden_dim 
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, hidden_size, num_heads, bias=use_bias))

        # Hidden layers:  hidden_size * num_heads -> hidden_size * num_heads
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_size * num_heads, hidden_size, num_heads, bias=use_bias))

        # Final GAT layer: Maps hidden_dim (hidden_size * num_heads) to num_classes with 1 head
        self.layers.append(GATConv(hidden_size * num_heads, num_classes, 1, bias=use_bias))

    def forward(self, graph, features):
        x = features
        for i, layer in enumerate(self.layers):
            x = layer(graph, x)
            if i != len(self.layers) - 1:  # Apply ReLU and flatten output from multiple heads for all but the last layer
                x = torch.relu(x)
                x = x.view(x.size(0), -1)  # Flatten output from multiple heads
                x = self.dropout(x)
        return x.squeeze() 
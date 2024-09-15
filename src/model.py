import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, node_features, hidden_dim, num_classes, dropout, num_layers, use_bias=True):
        """
        Args:
            node_features (int): Number of features per node.
            hidden_dim (int): Number of hidden units in each GCN layer.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
            num_layers (int): Number of GCN layers.
            use_bias (bool): Whether to use bias in the layers.
        """
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(node_features, hidden_dim, use_bias))

        for _ in range(1, num_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim, use_bias))

        self.gcn_layers.append(GCNLayer(hidden_dim, num_classes, use_bias))

    def initialize_weights(self):
        for layer in self.gcn_layers:
            layer.initialize_weights()

    # forward pass 
    def forward(self, x, adj):
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adj)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x
    
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCNLayer, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(input_dim, output_dim))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(output_dim,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias

        return torch.sparse.mm(adj, x)
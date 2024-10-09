import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, num_heads, dropout=0, num_layers=3, use_bias=True):
        super(GAT, self).__init__()
        
        #num_heads = [4, 4, 6]
        
        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = use_bias
        
        # First layer
        self.gat_layers.append(
            GATConv(
                in_feats,
                hidden_dim,
                num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=F.elu,
                residual=True,
                bias=use_bias
            )
        )
        
        # Intermediate layers
        for i in range(1, num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=F.elu,
                    residual=True,
                    bias=use_bias
                )
            )
        
        # Last layer
        self.gat_layers.append(
            GATConv(
                hidden_dim * num_heads,
                num_classes,
                num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=None,
                residual=True,
                bias=use_bias
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == self.num_layers - 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h
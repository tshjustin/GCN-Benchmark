import torch
from torch import nn
import torch.nn.functional as F

class GAT(nn.Module):

    def __init__(self, in_features, n_hidden, n_heads, num_classes, concat=False, dropout=0.4, leaky_relu_slope=0.2):
        """ 
        Args:
            in_features (int): number of input features per node.
            n_hidden (int): output size of the first Graph Attention Layer.
            n_heads (int): number of attention heads in the first Graph Attention Layer.
            num_classes (int): number of classes to predict for each node.
            concat (bool, optional): Wether to concatinate attention heads or take an average over them for the
                output of the first Graph Attention Layer. Defaults to False.
            dropout (float, optional): dropout rate. Defaults to 0.4.
            leaky_relu_slope (float, optional): alpha (slope) of the leaky relu activation. Defaults to 0.2.
        """

        super(GAT, self).__init__()

        self.gat1 = GraphAttentionLayer(in_features=in_features, out_features=n_hidden, n_heads=n_heads, concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope )
        
        self.gat2 = GraphAttentionLayer(in_features=n_hidden, out_features=num_classes, n_heads=1, concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope )
        

    def forward(self, input_tensor: torch.Tensor , adj_mat: torch.Tensor):
        """
        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.
        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        x = self.gat1(input_tensor, adj_mat)
        x = F.elu(x) 
        x = self.gat2(x, adj_mat)

        return F.log_softmax(x, dim=1) 
    

class GraphAttentionLayer(nn.Module):
    """
    Args:
        in_features (int): number of input features per node.
        n_hidden (int): output size of the first Graph Attention Layer.
        n_heads (int): number of attention heads 
        num_classes (int): number of classes to predict for each node.
        concat (bool, optional): Wether to concatinate final attention heads
            output of the first Graph Attention Layer. Defaults to False.
        dropout (float, optional): dropout rate. Defaults to 0.4.
        leaky_relu_slope (float, optional): alpha (slope) of the leaky relu activation. Defaults to 0.2.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int, concat: bool = False, dropout: float = 0.4, leaky_relu_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()

        self.n_heads = n_heads 
        self.concat = concat 
        self.dropout = dropout

        if concat:
            self.out_features = out_features 
            assert out_features % n_heads == 0 
            self.n_hidden = out_features // n_heads

        else:
            self.n_hidden = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope) 
        self.softmax = nn.Softmax(dim=1) 

        self.reset_parameters() 


    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)
    

    def _get_attention_scores(self, h_transformed: torch.Tensor):
        """
        calculates the attention scores e_ij for all pairs of nodes (i, j) in the graph

        Args:
            h_transformed (torch.Tensor): Transformed feature matrix with shape (n_nodes, n_heads, n_hidden),
                where n_nodes is the number of nodes and out_features is the number of output features per node.

        Returns:
            torch.Tensor: Attention score matrix with shape (n_heads, n_nodes, n_nodes), where n_nodes is the number of nodes.
        """
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])
        e = source_scores + target_scores.mT
        return self.leakyrelu(e)

    def forward(self,  h: torch.Tensor, adj_mat: torch.Tensor):
        """
        Performs a graph attention layer operation.

        Args:
            h (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """
        n_nodes = h.shape[0]
        h_transformed = torch.mm(h, self.W)
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)
        h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)

        e = self._get_attention_scores(h_transformed)

        connectivity_mask = -9e16 * torch.ones_like(e)
        e = torch.where(adj_mat > 0, e, connectivity_mask) # masked attention scores

        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h_transformed)

        if self.concat:
            h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)
        else:
            h_prime = h_prime.mean(dim=0)

        return h_prime
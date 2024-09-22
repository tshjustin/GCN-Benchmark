import pandas as pd
import torch
import numpy as np
import scipy.sparse as sparse
import os

def normalize_adjacency(adj):
    """Normalize the adjacency matrix."""
    identity_matrix = sparse.eye(adj.shape[0])
    adj_with_self_loops = adj + identity_matrix

    # degree matrix
    node_degrees = np.array(adj_with_self_loops.sum(axis=1)).flatten()

    # inverse square root of degree matrix
    inv_sqrt_degrees = np.power(node_degrees, -0.5)
    inv_sqrt_degrees[np.isinf(inv_sqrt_degrees)] = 0.0  # Handle infinite values
    inv_sqrt_degrees[np.isnan(inv_sqrt_degrees)] = 0.0  # Handle NaN values

    # Create diagonal matrix
    inv_sqrt_degree_matrix = sparse.diags(inv_sqrt_degrees, dtype=np.float32)

    # Normalize
    normalized_adj = inv_sqrt_degree_matrix @ adj_with_self_loops @ inv_sqrt_degree_matrix
    return normalized_adj

def covert_to_pytorch_format(matrix):
    """Convert the sparse matrix to PyTorch format."""
    matrix_helper_coo = matrix.tocoo().astype('float32')
    data = torch.FloatTensor(matrix_helper_coo.data)
    rows = torch.LongTensor(matrix_helper_coo.row)
    cols = torch.LongTensor(matrix_helper_coo.col)
    indices = torch.vstack([rows, cols])

    shape = torch.Size(matrix_helper_coo.shape)
    matrix = torch.sparse.FloatTensor(indices, data, shape)
    return matrix

def create_adjacency_matrix(edges, num_nodes):
    """Create the adjacency matrix from the edges."""
    row_indices = edges[:, 0]
    col_indices = edges[:, 1]
    data = np.ones(len(row_indices), dtype=np.float32)

    adj_shape = (num_nodes, num_nodes)
    adj_matrix = sparse.coo_matrix((data, (row_indices, col_indices)), shape=adj_shape)

    # since it's an undirected graph
    adj_symmetrical = adj_matrix + adj_matrix.T.multiply(adj_matrix.T > adj_matrix)
    adj_normalized = normalize_adjacency(adj_symmetrical)
    return adj_normalized

def load_ppi_graph_from_csv(graph_id, data_dir):
    """Load features, labels, and adjacency matrix for a specific graph from CSV files."""
    # Load edges
    edges_file = os.path.join(data_dir, f"ppi_graph_{graph_id}_edges.csv")
    edges_df = pd.read_csv(edges_file)
    edges = edges_df.to_numpy()

    # Load node features
    nodes_file = os.path.join(data_dir, f"ppi_graph_{graph_id}_nodes.csv")
    nodes_df = pd.read_csv(nodes_file)
    node_features = nodes_df.iloc[:, 1:].to_numpy()  # Skip node_id, take features

    # Load node labels
    labels_file = os.path.join(data_dir, f"ppi_graph_{graph_id}_labels.csv")
    labels_df = pd.read_csv(labels_file)
    node_labels = labels_df.iloc[:, 1:].to_numpy()  # Skip node_id, take labels

    # Create adjacency matrix from edges
    num_nodes = node_features.shape[0]
    adj_normalized = create_adjacency_matrix(edges, num_nodes)

    # Convert to PyTorch tensors
    features = torch.FloatTensor(node_features)
    labels = torch.FloatTensor(node_labels)
    adj = covert_to_pytorch_format(adj_normalized)

    return features, labels, adj, edges


def load_ppi_data(dir):
    """
    Processes all PPI Graphs 

    Returns: 
    all_features: List of list of graph node features 
    all_labels: List of list of each graph labels 
    all_adjs: List of list of each graph adj matrix 
    all_edges: List of list each graph edge 
    """

    print("Loading PPI Dataset")
    edge_files = [f for f in os.listdir(dir) if 'edges' in f]
    graph_ids = sorted([int(f.split('_')[2]) for f in edge_files])

    all_features = []
    all_labels = []
    all_adjs = []
    all_edges = []

    for graph_id in graph_ids:
        print(f"Processing graph {graph_id}")
        features, labels, adj, edges_ordered = load_ppi_graph_from_csv(graph_id, dir)

        all_features.append(features)
        all_labels.append(labels)
        all_adjs.append(adj)
        all_edges.append(edges_ordered)

    print("Finished loading PPI Dataset")
    return all_features, all_labels, all_adjs, all_edges
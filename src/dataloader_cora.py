import torch
import numpy as np
import scipy.sparse as sparse

def train_test_val_split(labels, num_classes, config):
    """Split to TTV proportional to class size to prevent imbalance"""
    classes = [i for i in range(num_classes)]
    train_set = []
    validation_set = []
    test_set = []

    total_samples = len(labels)
    total_train_size = int(total_samples * config.train_proportion)
    total_val_size = int(total_samples * config.validation_proportion)
    total_test_size = total_samples - total_train_size - total_val_size

    # Proportionally assign training samples from each class
    for class_label in classes:
        
        target_indices = torch.nonzero(labels == class_label, as_tuple=False).tolist()
        num_class_samples = len(target_indices)
        proportion_of_train_set = int(np.floor(num_class_samples * (config.train_proportion))) 
        train_set += [ind[0] for ind in target_indices[:proportion_of_train_set]]

    # Remainder for validation and test 
    validation_test_set = [ind for ind in range(len(labels)) if ind not in train_set]

    # Same as above
    for class_label in classes:
        target_indices = [ind for ind in validation_test_set if labels[ind] == class_label]
        num_class_samples = len(target_indices)
        proportion_of_val_set = int(np.floor(num_class_samples * (config.validation_proportion / (config.validation_proportion + config.test_proportion))))
        proportion_of_test_set = num_class_samples - proportion_of_val_set

        validation_set += target_indices[:proportion_of_val_set]
        test_set += target_indices[proportion_of_val_set:proportion_of_val_set + proportion_of_test_set]

    return train_set, validation_set, test_set

def map_labels(labels):
    "Map string labels to a numeric value"
    unique = sorted(set(labels)) 
    label_map = {label: idx for idx, label in enumerate(unique)}
    labels = np.array([label_map[label] for label in labels])
    return labels


def normalize_adjacency(adj):
    """normalization of adj matrix."""
    identity_matrix = sparse.eye(adj.shape[0])
    adj_with_self_loops = adj + identity_matrix

    # degree matrix
    node_degrees = np.array(adj_with_self_loops.sum(axis=1)).flatten()

    # inverse square root of  degree matrix
    inv_sqrt_degrees = np.power(node_degrees, -0.5)
    inv_sqrt_degrees[np.isinf(inv_sqrt_degrees)] = 0.0  # Handle infinite values
    inv_sqrt_degrees[np.isnan(inv_sqrt_degrees)] = 0.0  # Handle NaN values

    # Create diagonal matrix
    inv_sqrt_degree_matrix = sparse.diags(inv_sqrt_degrees, dtype=np.float32)

    # Normalize
    normalized_adj = inv_sqrt_degree_matrix @ adj_with_self_loops @ inv_sqrt_degree_matrix
    return normalized_adj

def reorder_node_id(node_id):
    """Reorder the random node indexes to a sequential one"""
    ids_ordered = {} # node_id : order
    for order, raw_id in enumerate(node_id):  
        ids_ordered[raw_id] = order

    return ids_ordered

def create_adjacency_matrix(edges_ordered, labels_enumerated): 
    """Creates the adjacency matrix"""
    num_edges = edges_ordered.shape[0] # number edges 

    row_indices = edges_ordered[:, 0] # source 
    col_indices = edges_ordered[:, 1] # target 

    data = np.ones(num_edges, dtype=np.float32) # weights 
    adj_shape = (labels_enumerated.shape[0], labels_enumerated.shape[0]) # dimension of matrix 
    adj_mat = sparse.coo_matrix((data, (row_indices, col_indices)), shape=adj_shape)

    # since undirected graph 
    adj_symmetrical = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat)
    adj_normalized = normalize_adjacency(adj_symmetrical)
    return adj_normalized

def covert_to_pytorch_format(matrix):
    """
    Converts to pytorch format for computation after loading sparse to memory
    """
    matrix_helper_coo = matrix.tocoo().astype('float32')
    data = torch.FloatTensor(matrix_helper_coo.data)
    rows = torch.LongTensor(matrix_helper_coo.row)
    cols = torch.LongTensor(matrix_helper_coo.col)
    indices = torch.vstack([rows, cols])

    shape = torch.Size(matrix_helper_coo.shape)
    matrix = torch.sparse.FloatTensor(indices, data, shape)
    return matrix

def load_cora_data(config):
    """Loads graph and put to sparse form for efficiancy"""

    print("Loading CORA Dataset")

    # loading node information  
    raw_data = np.genfromtxt(config.nodes_path, dtype="str")
    node_id = raw_data[:, 0].astype('int32')  # unique identifier of each node
    node_labels = raw_data[:, -1]

    # integer label tagging 
    labels_enumerated = map_labels(node_labels)  

    # sparse matrix representation 
    node_features = sparse.csr_matrix(raw_data[:, 1:-1], dtype="float32") # take the word features from w1 - w1433 
    
    # redefine ids 
    ordered_ids = reorder_node_id(node_id)
    
    # loading edges information & transform nodeids to match ordered one 
    raw_edges_data = np.genfromtxt(config.edges_path, dtype="int32")

    edges_ordered_list = []
    for edge in raw_edges_data:
        ordered_edge = [ordered_ids[edge[0]], ordered_ids[edge[1]]]
        edges_ordered_list.append(ordered_edge)

    edges_ordered = np.array(edges_ordered_list, dtype='int32') # example o/p of edges: array([[0, 1], [1, 2]]

    adj_normalized = create_adjacency_matrix(edges_ordered, labels_enumerated)
    features = torch.FloatTensor(node_features.toarray())
    labels = torch.LongTensor(labels_enumerated)
    adj = covert_to_pytorch_format(adj_normalized)

    print("Loaded CORA Dataset")

    return features, labels, adj, edges_ordered
import torch
import numpy as np
import scipy.sparse as sparse

def train_test_val_split(labels, num_classes, config):
    
    classes = [i for i in range(num_classes)]
    train_set = []

    # take 20 samples from each class 
    for class_label in classes:
        target_indices = torch.nonzero(labels == class_label, as_tuple=False).tolist()
        train_set += [ind[0] for ind in target_indices[:config.train_size_per_class]]

    validation_test_set = [ind for ind in range(len(labels)) if ind not in train_set]

    # remainder split 
    validation_set = validation_test_set[:config.validation_size]
    test_set = validation_test_set[config.validation_size:config.validation_size+config.test_size]

    return train_set, validation_set, test_set


def enumerate_labels(labels):
    """ Converts the labels from the original
        string form to the integer [0:MaxLabels-1]
    """
    unique = sorted(set(labels)) 
    label_map = {label: idx for idx, label in enumerate(unique)}
    labels = np.array([label_map[label] for label in labels])
    return labels


def normalize_adjacency(adj):
    """Normalizes the matrix"""
    adj = adj + sparse.eye(adj.shape[0])

    node_degrees = np.array(adj.sum(1))
    node_degrees = np.power(node_degrees, -0.5).flatten()
    node_degrees[np.isinf(node_degrees)] = 0.0
    node_degrees[np.isnan(node_degrees)] = 0.0
    degree_matrix = sparse.diags(node_degrees, dtype=np.float32)

    adj = degree_matrix @ adj @ degree_matrix
    return adj


def convert_scipy_to_torch_sparse(matrix):
    matrix_helper_coo = matrix.tocoo().astype('float32')
    data = torch.FloatTensor(matrix_helper_coo.data)
    rows = torch.LongTensor(matrix_helper_coo.row)
    cols = torch.LongTensor(matrix_helper_coo.col)
    indices = torch.vstack([rows, cols])

    shape = torch.Size(matrix_helper_coo.shape)
    matrix = torch.sparse.FloatTensor(indices, data, shape)
    return matrix


def load_data(config):
    """Loads graph and put to sparse form for efficiancy"""

    print("Loading Datase")

    raw_nodes_data = np.genfromtxt(config.nodes_path, dtype="str")
    raw_node_ids = raw_nodes_data[:, 0].astype('int32')  # unique identifier of each node
    raw_node_labels = raw_nodes_data[:, -1]
    labels_enumerated = enumerate_labels(raw_node_labels)  # target labels as integers
    node_features = sparse.csr_matrix(raw_nodes_data[:, 1:-1], dtype="float32")

    # load structure 
    ids_ordered = {raw_id: order for order, raw_id in enumerate(raw_node_ids)}
    raw_edges_data = np.genfromtxt(config.edges_path, dtype="int32")
    edges_ordered = np.array(list(map(ids_ordered.get, raw_edges_data.flatten())),
                             dtype='int32').reshape(raw_edges_data.shape)

    # load adjaency matrix    
    adj = sparse.coo_matrix((np.ones(edges_ordered.shape[0]), (edges_ordered[:, 0], edges_ordered[:, 1])),
                            shape=(labels_enumerated.shape[0], labels_enumerated.shape[0]),
                            dtype=np.float32)
    
    # Make the adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj)
    adj = normalize_adjacency(adj)

    # change to pytorch format 
    features = torch.FloatTensor(node_features.toarray())
    labels = torch.LongTensor(labels_enumerated)
    adj = convert_scipy_to_torch_sparse(adj)

    print("Dataset loaded")

    return features, labels, adj, edges_ordered
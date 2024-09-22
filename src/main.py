from model.GCN import GCN
from model.GAT import GAT 
from dataloader_CORA import *
from dataloader_PPI import load_ppi_data 
from evaluation_PPI import train_PPI, evaluate_PPI
from evaluation_CORA import train_CORA, evaluate_CORA
from visuals import * 
from args import config

"""
Would need a major refactoring of this portion to accomodate for PPI data 
"""

if __name__ == "__main__": 
    
    if config.model == "GCN":
        features, labels, adj, edges = load_data(config)
        
        NUM_CLASSES = int(labels.max().item() + 1)

        train_set_ind, val_set_ind, test_set_ind = train_test_val_split(labels, NUM_CLASSES, config)

        model = GCN(features.shape[1], config.hidden_dim, NUM_CLASSES, config.dropout, config.num_layers, config.use_bias)

        train_acc, train_loss, val_acc, val_loss = train_CORA(model, features, labels, adj, train_set_ind, val_set_ind, config)
        prediction = evaluate_CORA(model, features, labels, adj, test_set_ind, config)

        print("Number of GCN-Layers:", config.num_layers)
        print("Dropout Rate:", config.dropout)
        print("Learning Rate:", config.lr, "Decay Rate", config.weight_decay)

        visualize_embedding_tSNE(labels, prediction, NUM_CLASSES)
        visualize_train_performance(train_acc, train_loss)
        visualize_val_performance(val_acc, val_loss)

    elif config.model == "GAT":
        features, labels, adj, edges = load_data(config)
        
        NUM_CLASSES = int(labels.max().item() + 1)

        train_set_ind, val_set_ind, test_set_ind = train_test_val_split(labels, NUM_CLASSES)

        model = GAT(features.shape[1], config.n_hidden_units, config.n_heads, 
                    NUM_CLASSES, config.num_layers, config.concat, config.dropout, config.leaky_relu_slope)

        train_acc, train_loss, val_acc, val_loss = train_CORA(model, features, labels, adj, train_set_ind, val_set_ind, config)
        prediction = evaluate_CORA(model, features, labels, adj, test_set_ind, config)

        print("Number of GCN-Layers:", config.num_layers)
        print("Dropout Rate:", config.dropout)
        print("Learning Rate:", config.lr, "Decay Rate", config.weight_decay)

        visualize_embedding_tSNE(labels, prediction, NUM_CLASSES)
        visualize_train_performance(train_acc, train_loss)
        visualize_val_performance(val_acc, val_loss)
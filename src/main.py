from model.GCN import GCN
from model.GAT import GAT 
from dataloader_CORA import *
from dataloader_PPI import load_ppi_data 
from evaluation_PPI import train_PPI, evaluate_PPI
from evaluation_CORA import train_CORA, evaluate_CORA
from visuals import * 
from args import config

if __name__ == "__main__": 
    
    if config.dataset == "CORA":
        features, labels, adj, edges = load_cora_data(config)
        
        NUM_CLASSES = int(labels.max().item() + 1)

        train_set_ind, val_set_ind, test_set_ind = train_test_val_split(labels, NUM_CLASSES, config)
        
        if config.model == "GCN":
            model = GCN(features.shape[1], config.hidden_dim, NUM_CLASSES, config.dropout, config.num_layers, config.use_bias)

        elif config.model == "GAT": 
            model = GAT(features.shape[1], config.n_hidden_units, config.n_heads, NUM_CLASSES, config.num_layers, config.concat, config.dropout, config.leaky_relu_slope)

        train_acc, train_loss, val_acc, val_loss = train_CORA(model, features, labels, adj, train_set_ind, val_set_ind, config)
        prediction = evaluate_CORA(model, features, labels, adj, test_set_ind, config)

    elif config.dataset == "PPI": 
        
        train_features, train_labels, train_adjs, _ = load_ppi_data(config.train_dir)
        val_features, val_labels, val_adjs, _ = load_ppi_data(config.val_dir)
        test_features, test_labels, test_adjs, _ = load_ppi_data(config.test_dir)

        NUM_CLASSES = train_labels[0].shape[1]

        if config.model == "GCN":
            model = GCN(train_features[0].shape[1], config.hidden_dim, NUM_CLASSES, config.dropout, config.num_layers, config.use_bias)

        elif config.model == "GAT":
            pass

        train_f1_list, train_loss_list, val_f1_list, val_loss_list = train_PPI(model, train_features, train_labels, train_adjs, val_features, val_labels, val_adjs, config)
        evaluate_PPI(model, test_features, test_labels, test_adjs)

    
    # Meta Data 
    # print("Number of Layers:", config.num_layers)
    # print("Dropout Rate:", config.dropout)
    # print("Learning Rate:", config.lr, "Decay Rate", config.weight_decay)
    # visualize_embedding_tSNE(labels, prediction, NUM_CLASSES)
    # visualize_train_performance(train_acc, train_loss)
    # visualize_val_performance(val_acc, val_loss)

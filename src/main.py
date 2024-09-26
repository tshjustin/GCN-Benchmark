from model.GCN import GCN
from model.GAT import GAT 
from dataloader import *
from evaluation import *
from visuals import * 
from args import config

if __name__ == "__main__": 
    
    if config.dataset == "CORA":
        features, labels, adj, edges = load_cora_data(config)

    elif config.dataset == "SQU":
        features, labels, adj, edges = load_squirrel_data(config)

    NUM_CLASSES = int(labels.max().item() + 1)

    train_set_ind, val_set_ind, test_set_ind = train_test_val_split(labels, NUM_CLASSES, config)
        
    if config.model == "GCN":
        model = GCN(features.shape[1], config.hidden_dim, NUM_CLASSES, config.dropout, config.num_layers, config.use_bias)

    elif config.model == "GAT": 
        model = GAT(features.shape[1], config.n_hidden_units, config.n_heads, NUM_CLASSES, config.num_layers, config.concat, config.dropout, config.leaky_relu_slope)

    train_acc, train_loss, val_acc, val_loss = train(model, features, labels, adj, train_set_ind, val_set_ind, config)
    prediction = evaluate(model, features, labels, adj, test_set_ind)

    
    # Meta Data 
    # print("Number of Layers:", config.num_layers)
    # print("Dropout Rate:", config.dropout)
    # print("Learning Rate:", config.lr, "Decay Rate", config.weight_decay)
    # visualize_embedding_tSNE(labels, prediction, NUM_CLASSES)
    # visualize_train_performance(train_acc, train_loss)
    # visualize_val_performance(val_acc, val_loss)

from model import GCN
from dataloader import *
from args import config
from evaluation import *
from visuals import * 

if __name__ == "__main__":
     
    features, labels, adj, edges = load_data(config)
    
    NUM_CLASSES = int(labels.max().item() + 1)

    train_set_ind, val_set_ind, test_set_ind = train_test_val_split(labels, NUM_CLASSES, config)

    model = GCN(features.shape[1], config.hidden_dim, NUM_CLASSES, config.dropout, config.num_layers, config.use_bias)

    train_acc, train_loss, val_acc, val_loss = train(model, features, labels, adj, train_set_ind, val_set_ind, config)
    prediction = evaluate_on_test(model, features, labels, adj, test_set_ind, config)

    print("Number of GCN-Layers:", config.num_layers)
    print("Dropout Rate:", config.dropout)
    print("Learning Rate:", config.lr, "Decay Rate", config.weight_decay)

    visualize_embedding_tSNE(labels, prediction, NUM_CLASSES)
    visualize_performance(train_acc, train_loss, val_acc, val_loss)
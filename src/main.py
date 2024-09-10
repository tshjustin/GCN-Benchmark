from model import GCN
from dataloader import *
from args import config
from evaluation import *

if __name__ == "__main__":
    
    # load 
    features, labels, adj, edges = load_data(config)
    
    # visuals 
    NUM_CLASSES = int(labels.max().item() + 1)

    train_set_ind, val_set_ind, test_set_ind = train_test_val_split(labels, NUM_CLASSES, config)

    model = GCN(features.shape[1], config.hidden_dim, NUM_CLASSES, config.dropout, config.use_bias)

    val_acc, val_loss = train(model, features, labels, adj, train_set_ind, val_set_ind, config)
    out_features = evaluate_on_test(model, features, labels, adj, test_set_ind, config)


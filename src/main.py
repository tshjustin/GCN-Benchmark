import dgl
from model.GCN import GCN 
from model.GAT import GAT
from args import config
from evaluate import train_loop, setup_optimization
from visuals import * 
import itertools
import torch 

results = {} 
combinations = itertools.product(config.num_layers, config.lr, config.hidden_dim, config.dropout, config.num_heads)

if __name__ == "__main__":

    # Dataset Choice
    if config.dataset == "CORA":
        data = dgl.data.CoraGraphDataset()
        graph = data[0]

        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
    
    elif config.dataset == "SQU":
        data = dgl.data.SquirrelDataset()
        graph = data[0]

        train_mask = graph.ndata['train_mask'][:, 0].bool()
        val_mask = graph.ndata['val_mask'][:, 0].bool()
        test_mask = graph.ndata['test_mask'][:, 0].bool()

    features = graph.ndata['feat']  
    labels = graph.ndata['label']  
    in_feats = features.shape[1]  # Get input size 
    num_classes = data.num_classes # Get the number of classes 

    for num_layers, lr, hidden_dim, dropout, n_heads in combinations: # Run all combinations 
        if config.model == "GCN":
            model = GCN(in_feats, hidden_dim, num_classes, dropout, num_layers, config.use_bias)
            
        elif config.model == "GAT":
            model = GAT(in_feats, hidden_dim, num_classes, n_heads, dropout, num_layers, config.use_bias)
            
        optimizer, criterion = setup_optimization(model, lr)
    
        # Training of Model 
        print(f"Model: {config.model}, Dataset: {config.dataset}, Layers: {num_layers}, Hidden_Dim: {hidden_dim}, Dropout: {dropout}")
        train_losses, train_accuracies, val_losses, val_accuracies = train_loop(model, optimizer, criterion, graph, features, labels, train_mask, val_mask, config.epochs)

        # Storing of results 
        key = f"{num_layers}_layers_lr_{lr}_hidden_{hidden_dim}_dropout_{dropout}"
        results[key] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }

    # Get Configuration Settings 
    print("Configurations:")
    for arg in vars(config):
        print(f"{arg}: {getattr(config, arg)}")

    with torch.no_grad():
        out_features = model(graph, features)

    visualize_train_performance(results)
    visualize_val_performance(results)
    visualize_embedding_tSNE(labels, out_features, num_classes)
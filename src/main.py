import dgl
from model.GCN import GCN 
from model.GAT import GAT
from args import config
from evaluate import train_loop, setup_optimization, evaluate
from visuals import * 
import itertools
import torch 

results = {} 
combinations = itertools.product(config.num_layers, config.hidden_dim, config.lr, config.dropout)

results = {} 
combinations = itertools.product(config.num_layers, config.hidden_dim, config.lr, config.dropout)

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
    num_classes = data.num_classes  # Get the number of classes 

    for num_layer, hidden_dim, lr, dropout in combinations:
        print(f"Training model with {num_layer} layers, lr={lr}, hidden_dim={hidden_dim}, dropout={dropout}")

        if config.model == "GCN":
            model = GCN(in_feats, hidden_dim, num_classes, dropout, num_layer, config.use_bias)
            key = f"{num_layer}_layers_lr_{lr}_hidden_{hidden_dim}_dropout_{dropout}_GCN"

            optimizer, criterion = setup_optimization(model, lr)

            # Train and validate 
            train_losses, train_accuracies, val_losses, val_accuracies = train_loop(
                model, optimizer, criterion, graph, features, labels, train_mask, val_mask, config.epochs
            )

            # Evaluating the model 
            test_acc, test_loss = evaluate(model, test_mask, graph, features, labels, criterion)
            print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

            results[key] = {
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'test_loss': test_loss,
                'test_acc': test_acc
            }

        # Initialize and train GAT with varying n_heads
        elif config.model == "GAT":
            for n_head in config.num_heads:
                print(f"Training GAT with {n_head} heads")

                model = GAT(in_feats, hidden_dim, num_classes, n_head, dropout, num_layer, config.use_bias)
                key = f"{num_layer}_layers_lr_{lr}_hidden_{hidden_dim}_dropout_{dropout}_n_heads_{n_head}_GAT"

                optimizer, criterion = setup_optimization(model, lr)

                # Train and validate
                train_losses, train_accuracies, val_losses, val_accuracies = train_loop(
                    model, optimizer, criterion, graph, features, labels, train_mask, val_mask, config.epochs
                )

                test_acc, test_loss = evaluate(model, test_mask, graph, features, labels, criterion)
                print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

                results[key] = {
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                    'test_loss': test_loss,
                    'test_acc': test_acc
                }

    # Print Configuration Settings 
    print("Configurations:")
    for arg in vars(config):
        print(f"{arg}: {getattr(config, arg)}")

    # Visualize Training and Validation Performance
    visualize_train_performance(results)
    visualize_val_performance(results)

    # Embedding Visualization
    with torch.no_grad():
        out_features = model(graph, features)
    visualize_embedding_tSNE(labels, out_features, num_classes)
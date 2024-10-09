import dgl
import torch 
import itertools
from torch.utils.data import DataLoader
from model.GCN import GCN 
from model.GAT import GAT
from model.GSage import GraphSAGE
from args import config
from evaluate import train_loop, setup_optimization, evaluate
from evalutate_PPI import train_loop_ppi, setup_optimization_ppi, evaluate_ppi, collate_fn, train_loop_ppi_f1, evaluate_ppi_f1
from visuals import *
from dgl import AddSelfLoop
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = {} 
combinations = itertools.product(config.num_layers, config.hidden_dim, config.lr, config.dropout)

def save_results_to_csv(results, filename='results.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Model', 'Num Layers', 'Learning Rate', 'Hidden Dim', 'Dropout', 'Num Heads', 'Train Losses', 'Train Accuracies', 'Val Losses', 'Val Accuracies', 'Test Loss', 'Test Acc'])
        
        for result in results.values():
            writer.writerow([
                result['dataset'],
                result['model_type'], result['num_layers'], result['lr'], result['hidden_dim'], result['dropout'], result['num_heads'],
                result['train_losses'][-1], result['train_accuracies'],
                result['val_losses'], result['val_accuracies'],
                result['test_loss'], result['test_acc']
            ])

if __name__ == "__main__":
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    # Dataset Choice
    if config.dataset == "CORA":
        data = dgl.data.CoraGraphDataset(transform=transform)
        graph = data[0].to(device)

        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
    
    elif config.dataset == "SQU":
        data = dgl.data.SquirrelDataset()
        graph = data[0].to(device)
        train_mask = graph.ndata['train_mask'][:, 0].bool()
        val_mask = graph.ndata['val_mask'][:, 0].bool()
        test_mask = graph.ndata['test_mask'][:, 0].bool()

    elif config.dataset == "PPI": 
        train_dataset = dgl.data.PPIDataset(mode='train') 
        valid_dataset = dgl.data.PPIDataset(mode='valid')  
        test_dataset = dgl.data.PPIDataset(mode='test')  

        # Setting to the first graph for the sole purpose of feature shapes for the model
        first_graph = train_dataset[0].to(device)
        graph = first_graph

    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)
    in_feats = features.shape[1]  

    # Since PPI is a Multi-Label Classification Problem
    if config.dataset == "PPI":
        num_classes = labels.shape[1]  
    else:
        num_classes = data.num_classes 

    for num_layer, hidden_dim, lr, dropout in combinations:
        print(f"Training model with {num_layer} layers, lr={lr}, hidden_dim={hidden_dim}, num_heads={config.num_heads}, dropout={dropout}")

        # ------------------ GCN MODEL ------------------ 
        if config.model == "GCN":
            model = GCN(in_feats, hidden_dim, num_classes, dropout, num_layer, config.use_bias).to(device)  # Move model to device
            key = f"{num_layer}_layers_lr_{lr}_hidden_{hidden_dim}_dropout_{dropout}_GCN"

            optimizer, criterion = setup_optimization(model, lr)

            # For PPI, use DataLoader and process batches of graphs
            if config.dataset == "PPI":
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

                optimizer, criterion = setup_optimization_ppi(model, lr)
                
                train_losses, train_accuracies, val_losses, val_accuracies = train_loop_ppi(
                    model, optimizer, criterion, train_loader, val_loader, config.epochs, device
                )

                test_acc, test_loss = evaluate_ppi_f1(model, test_loader, criterion, device)
                print(f"Test Loss: {test_loss:.4f} | Test Accuracy : {test_acc:.4f}")

            else: 
                train_losses, train_accuracies, val_losses, val_accuracies = train_loop(
                    model, optimizer, criterion, graph, features, labels, train_mask, val_mask, config.epochs, device
                )

                test_acc, test_loss = evaluate(model, test_mask, graph, features, labels, criterion, device)
                print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

            results[key] = {
                'dataset': config.dataset,
                'model_type': 'GCN',
                'num_layers': num_layer,
                'lr': lr,
                'hidden_dim': hidden_dim,
                'dropout': dropout,
                'num_heads': '',
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'test_loss': test_loss,
                'test_acc': test_acc
            }

        # ------------------ GAT MODEL ------------------ 
        elif config.model == "GAT":
            for n_head in config.num_heads:
                print(f"Training GAT with {n_head} heads")

                model = GAT(in_feats, hidden_dim, num_classes, n_head, dropout, num_layer, config.use_bias).to(device)
                key = f"{num_layer}_layers_lr_{lr}_hidden_{hidden_dim}_dropout_{dropout}_n_heads_{n_head}_GAT"

                optimizer, criterion = setup_optimization(model, lr)

                # For PPI, use DataLoader and process batches of graphs
                if config.dataset == "PPI":
                    optimizer, criterion = setup_optimization_ppi(model, lr)
                    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
                    val_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
                    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

                    train_losses, train_accuracies, val_losses, val_accuracies = train_loop_ppi(
                        model, optimizer, criterion, train_loader, val_loader, config.epochs, device
                    )

                    test_acc, test_loss = evaluate_ppi_f1(model, test_loader, criterion, device)
                    print(f"Test Loss: {test_loss:.4f} | Test Accuracy : {test_acc:.4f}")

                else: 
                    train_losses, train_accuracies, val_losses, val_accuracies = train_loop(
                        model, optimizer, criterion, graph, features, labels, train_mask, val_mask, config.epochs, device
                    )

                    test_acc, test_loss = evaluate(model, test_mask, graph, features, labels, criterion, device)
                    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

                results[key] = {
                    'dataset': config.dataset,
                    'model_type': 'GAT',
                    'num_layers': num_layer,
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                    'dropout': dropout,
                    'num_heads': n_head,
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                    'test_loss': test_loss,
                    'test_acc': test_acc
                }

        # ------------------ GraphSAGE MODEL ------------------ 
        elif config.model == "GraphSAGE":
            model = GraphSAGE(in_feats, hidden_dim, num_classes, dropout, num_layer, config.use_bias).to(device)  # Move model to device
            key = f"{num_layer}_layers_lr_{lr}_hidden_{hidden_dim}_dropout_{dropout}_GraphSAGE"

            optimizer, criterion = setup_optimization(model, lr)

            # For PPI, use DataLoader and process batches of graphs
            if config.dataset == "PPI":
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

                optimizer, criterion = setup_optimization_ppi(model, lr)
                
                train_losses, train_accuracies, val_losses, val_accuracies = train_loop_ppi(
                    model, optimizer, criterion, train_loader, val_loader, config.epochs, device
                )

                test_acc, test_loss = evaluate_ppi_f1(model, test_loader, criterion, device)
                print(f"Test Loss: {test_loss:.4f} | Test Accuracy : {test_acc:.4f}")

            else: 
                train_losses, train_accuracies, val_losses, val_accuracies = train_loop(
                    model, optimizer, criterion, graph, features, labels, train_mask, val_mask, config.epochs, device
                )

                test_acc, test_loss = evaluate(model, test_mask, graph, features, labels, criterion, device)
                print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

            results[key] = {
                'dataset': config.dataset,
                'model_type': 'GraphSAGE',
                'num_layers': num_layer,
                'lr': lr,
                'hidden_dim': hidden_dim,
                'dropout': dropout,
                'num_heads': '',
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

    if config.dataset == "PPI": 
        visualize_embedding_tSNE_multilabel(labels, out_features, num_classes)
    else:
        visualize_embedding_tSNE(labels, out_features, num_classes)

    # Save results to CSV
    save_results_to_csv(results)
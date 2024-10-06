import dgl
import torch
import time
import itertools
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.GCN import GCN
from model.GAT import GAT
from args import config
from evaluate import train_loop, setup_optimization, evaluate
from evalutate_PPI import train_loop_ppi, setup_optimization_ppi, evaluate_ppi, collate_fn

def compare_methods():
    results = {}
    combinations = itertools.product(config.num_layers, config.hidden_dim, config.lr, config.dropout) # Not really needed since we want to keep the hyperparam the same while comparing
    
    for num_layer, hidden_dim, lr, dropout in combinations: 
        print(f"Comparing GCN and GAT with {num_layer} layers, lr={lr}, hidden_dim={hidden_dim}, dropout={dropout}")

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

        elif config.dataset == "PPI":
            train_dataset = dgl.data.PPIDataset(mode='train')
            valid_dataset = dgl.data.PPIDataset(mode='valid')
            test_dataset = dgl.data.PPIDataset(mode='test')
            first_graph = train_dataset[0]
            graph = first_graph

        features = graph.ndata['feat']
        labels = graph.ndata['label']
        in_feats = features.shape[1]

        if config.dataset == "PPI":
            num_classes = labels.shape[1]  # Multi-label classification
        else:
            num_classes = data.num_classes  # Single-label classification
        
        # ------------- Track Results -------------
        models = {'GCN': GCN, 'GAT': GAT}
        accuracy_results = {'GCN': [], 'GAT': []}
        time_results = {'GCN': [], 'GAT': []}

        print(f"Dataset {config.dataset}")
        
        for model_name, ModelClass in models.items():
            print(f"Training {model_name}...")

            if model_name == "GAT":
                for n_heads in config.num_heads:
                    model = ModelClass(in_feats, hidden_dim, num_classes, n_heads, dropout, num_layer, config.use_bias)
            else:
                model = ModelClass(in_feats, hidden_dim, num_classes, dropout, num_layer, config.use_bias)

            optimizer, criterion = setup_optimization_ppi(model, lr) if config.dataset == "PPI" else setup_optimization(model, lr)

            start_time = time.time()  # Record start time

            if config.dataset == "PPI":
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

                train_losses, train_accuracies, val_losses, val_accuracies = train_loop_ppi(
                    model, optimizer, criterion, train_loader, val_loader, config.epochs
                )

                test_acc, test_loss = evaluate_ppi(model, test_loader, criterion)
            else:
                train_losses, train_accuracies, val_losses, val_accuracies = train_loop(
                    model, optimizer, criterion, graph, features, labels, train_mask, val_mask, config.epochs
                )

                test_acc, test_loss = evaluate(model, test_mask, graph, features, labels, criterion)

            end_time = time.time()  # Record end time
            time_taken = end_time - start_time

            # Store accuracy and time results
            accuracy_results[model_name].append(test_acc)
            time_results[model_name].append(time_taken)

            print(f"{model_name} - Test Accuracy: {test_acc:.4f} | Time Taken: {time_taken:.2f} seconds")

        # ----------- Plot Comparison -----------
        plot_comparison(accuracy_results, time_results)

def plot_comparison(accuracy_results, time_results):
    """Plot the comparison of accuracy and time taken for GCN and GAT."""
    labels = list(accuracy_results.keys())  # ['GCN', 'GAT']

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for model_name in labels:
        plt.bar(model_name, accuracy_results[model_name][-1], label=model_name)
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Comparison (GCN vs GAT)')

    plt.subplot(1, 2, 2)
    for model_name in labels:
        plt.bar(model_name, time_results[model_name][-1], label=model_name)
    plt.ylabel('Time Taken (seconds)')
    plt.title('Training Time Comparison (GCN vs GAT)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_methods()

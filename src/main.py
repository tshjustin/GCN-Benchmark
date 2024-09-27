import torch
import dgl
from model.GCN import GCN 
from model.GAT import GAT
from args import config
from evaluate import train, evaluate, setup_optimization

if __name__ == "__main__":
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

    if config.model == "GCN":
        model = GCN(in_feats, config.hidden_dim, num_classes, config.dropout, config.num_layers, config.use_bias)
        
    elif config.model == "GAT":
        model = GAT(in_feats, config.hidden_dim, num_classes, config.n_heads, config.dropout, config.num_layers, config.use_bias)
    
    optimizer, criterion = setup_optimization(model, config.lr)
    
    # Storing of Results 
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training of Model 
    print(f"Model {config.model} is used with {config.dataset} dataset")

    for epoch in range(config.epochs):
        # Train and store training loss and accuracy
        train_loss, train_acc = train(model, optimizer, criterion, graph, features, labels, train_mask)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_acc, val_loss = evaluate(model, val_mask, graph, features, labels, criterion)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | Training Loss: {train_loss:.4f} | Training Accuracy: {train_acc:.4f} | "
            f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

    test_acc, _ = evaluate(model, test_mask, graph, features, labels, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Get Configuration Settings 
    print("Configurations:")
    for arg in vars(config):
        print(f"{arg}: {getattr(config, arg)}")

    # # Training loop for GAT
    # for epoch in range(50):
    #     loss = train(gat_model, gat_optimizer, gat_criterion, graph, features, labels, train_mask)
    #     val_acc = evaluate(gat_model, val_mask, graph, features, labels)
    #     print(f"Epoch {epoch+1} | GAT Loss: {loss:.4f} | GAT Validation Accuracy: {val_acc:.4f}")

    # gat_test_acc = evaluate(gat_model, test_mask, graph, features, labels)
    # print(f"GAT Test Accuracy: {gat_test_acc:.4f}")

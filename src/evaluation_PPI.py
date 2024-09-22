import torch.nn as nn 
import torch 
from utils import f1 

def train_PPI(model, train_features, train_labels, train_adj, val_features, val_labels, val_adj, config):

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_loss_list = []
    train_f1_list = []
    validation_loss = []
    validation_f1 = []

    if config.early_termination:
        last_min_val_loss = float('inf')
        patience_counter = 0
        stopped_early = False

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        total_train_f1 = 0
        total_val_loss = 0
        total_val_f1 = 0

        # Loop over training graphs
        for graph_idx in range(len(train_features)):
            features = train_features[graph_idx]
            labels = train_labels[graph_idx]
            adj = train_adj[graph_idx]

            optimizer.zero_grad()

            # Forward pass 
            y_pred = model(features, adj)

            train_loss = criterion(y_pred, labels)
            train_f1 = f1(y_pred, labels)

            # Backpropagation
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()
            total_train_f1 += train_f1

        # Validation step
        model.eval()
        with torch.no_grad():
            for graph_idx in range(len(val_features)):
                val_features_graph = val_features[graph_idx]
                val_labels_graph = val_labels[graph_idx]
                val_adjs_graph = val_adj[graph_idx]

                y_pred_val = model(val_features_graph, val_adjs_graph)

                val_loss = criterion(y_pred_val, val_labels_graph)
                val_f1 = f1(y_pred_val, val_labels_graph)

                total_val_loss += val_loss.item()
                total_val_f1 += val_f1

        avg_train_loss = total_train_loss / len(train_features)
        avg_train_f1 = total_train_f1 / len(train_features)
        avg_val_loss = total_val_loss / len(val_features)
        avg_val_f1 = total_val_f1 / len(val_features)

        train_loss_list.append(avg_train_loss)
        train_f1_list.append(avg_train_f1)
        validation_loss.append(avg_val_loss)
        validation_f1.append(avg_val_f1)

        # Print epoch results
        print(" | ".join([f"Epoch: {epoch:4d}",
                          f"Train loss: {avg_train_loss:.3f}",
                          f"Train F1: {avg_train_f1:.3f}",
                          f"Val loss: {avg_val_loss:.3f}",
                          f"Val F1: {avg_val_f1:.3f}"
                          ]))

        # Early stopping check
        if config.early_termination:
            if avg_val_loss < last_min_val_loss:
                last_min_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == config.patience:
                    print(f"stopping epoch {epoch} with no improvement in validation loss")
                    stopped_early = True

        if stopped_early:
            break

    return train_f1_list, train_loss_list, validation_f1, validation_loss

def evaluate_PPI(model, features_list, labels_list, adjs_list):
    total_loss = 0
    total_f1 = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for graph_idx in range(len(features_list)):
                features = features_list[graph_idx]
                labels = labels_list[graph_idx]
                adj = adjs_list[graph_idx]

                output = model(features, adj)
                output_probs = torch.sigmoid(output)
                loss = criterion(output, labels)
                total_loss += loss.item()

                f1_score_graph = f1(output_probs, labels)
                total_f1 += f1_score_graph
        

    avg_f1_score = total_f1 / len(features_list)
    print(f"Test Loss: {total_loss:.5f}  |  Average F1-Score: {avg_f1_score:.5f}")
    return avg_f1_score, total_loss
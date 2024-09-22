import torch.nn as nn 
import torch 
from utils import f1 

def train_PPI(model, features, labels, adj, train_set_ind, val_set_ind, config):

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()  # Use binary cross-entropy loss for multi-label classification

    train_f1_list = []
    train_loss_list = []
    validation_f1 = []
    validation_loss = []

    if config.early_termination:
        last_min_val_loss = float('inf')
        patience_counter = 0
        stopped_early = False

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        model.train()

        y_pred = model(features, adj)
        train_loss = criterion(y_pred[train_set_ind], labels[train_set_ind])

        train_f1 = f1(y_pred[train_set_ind], labels[train_set_ind])

        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_loss = criterion(y_pred[val_set_ind], labels[val_set_ind])
            val_f1 = f1(y_pred[val_set_ind], labels[val_set_ind])

            train_loss_list.append(train_loss.item())
            train_f1_list.append(train_f1)
            validation_loss.append(val_loss.item())
            validation_f1.append(val_f1)

            if config.early_termination:
                if val_loss < last_min_val_loss:
                    last_min_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter == config.patience:
                        stopped_early = True
        
        print(" | ".join([f"Epoch: {epoch:4d}",
                        f"Train loss: {train_loss.item():.3f}",
                        f"Train F1: {train_f1:.3f}",
                        f"Val loss: {val_loss.item():.3f}",
                        f"Val F1: {val_f1:.3f}"
                        ]))

        if config.early_termination and stopped_early:
            break

    if config.early_termination and stopped_early:
        print(f"Early stopping at epoch: {epoch} due to no significant improvement.")

    return train_f1_list, train_loss_list, validation_f1, validation_loss

def evaluate_PPI(model, features_list, labels_list, adjs_list):
    """Buggy lol"""
    
    total_loss = 0
    total_f1 = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for graph_idx in range(len(features_list)):
                features = features_list[graph_idx]
                labels = labels_list[graph_idx]
                adj = adjs_list[graph_idx]

                output = model(features, adj)
                loss = criterion(output, labels)
                total_loss += loss.item()

                f1_score_graph = f1(output, labels)
                total_f1 += f1_score_graph
        

    avg_f1_score = total_f1 / len(features_list)
    print(f"Test Loss: {total_loss:.5f}  |  Average F1-Score: {avg_f1_score:.5f}")
    return avg_f1_score, total_loss
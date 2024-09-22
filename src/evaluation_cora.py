import torch
import torch.nn as nn
from utils import accuracy

def train_CORA(model, features, labels, adj, train_set_ind, val_set_ind, config):

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_acc_list = []
    train_loss_list = []
    validation_acc = []
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
        train_acc = accuracy(y_pred[train_set_ind], labels[train_set_ind])

        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_loss = criterion(y_pred[val_set_ind], labels[val_set_ind])
            val_acc = accuracy(y_pred[val_set_ind], labels[val_set_ind])

            # performance 
            train_loss_list.append(train_loss.item())
            train_acc_list.append(train_acc)
            validation_loss.append(val_loss.item())
            validation_acc.append(val_acc)

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
                        f"Train acc: {train_acc:.2f}",
                        f"Val loss: {val_loss.item():.3f}",
                        f"Val acc: {val_acc:.2f}"
                        ]))

        if config.early_termination and stopped_early:
            break

    if config.early_termination and stopped_early:
        print(f"Negligible model improvement. Stopped at epoch: {epoch}.")

    return train_acc_list, train_loss_list, validation_acc, validation_loss, 


def evaluate_CORA(model, features, labels, adj, test_ind):
    """Returns the logits for each nodes"""

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()
        y_pred = model(features, adj)
        test_loss = criterion(y_pred[test_ind], labels[test_ind])
        test_acc = accuracy(y_pred[test_ind], labels[test_ind]) # performs the argmax 

    print()
    print(f"Testing Accuracy loss: {test_loss:.5f}  |  Testing Accuracy: {test_acc:.5f}")
    return y_pred
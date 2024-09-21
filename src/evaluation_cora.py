import torch
import torch.nn as nn
from utils import * 
from sklearn.metrics import f1_score

def train(model, features, labels, adj, train_set_ind, val_set_ind, config):

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

        train_f1 = f1_score(labels[train_set_ind].cpu(), y_pred[train_set_ind].argmax(dim=1).cpu(), average='micro')

        with torch.no_grad():
            model.eval()
            val_loss = criterion(y_pred[val_set_ind], labels[val_set_ind])
            val_acc = accuracy(y_pred[val_set_ind], labels[val_set_ind])

            val_f1 = f1_score(labels[val_set_ind].cpu(), y_pred[val_set_ind].argmax(dim=1).cpu(), average='micro')

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
                        f"Train F1: {train_f1:.2f}",
                        f"Val loss: {val_loss.item():.3f}",
                        f"Val acc: {val_acc:.2f}",
                        f"Val F1: {val_f1:.2f}"]))

        if config.early_termination and stopped_early:
            break

    if config.early_termination and stopped_early:
        print(f"Negligible model improvement. Stopped at epoch: {epoch}.")

    return train_acc_list, train_loss_list, validation_acc, validation_loss, 


def evaluate_on_test(model, features, labels, adj, test_ind, config):

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()
        y_pred = model(features, adj)
        test_loss = criterion(y_pred[test_ind], labels[test_ind])
        test_acc = accuracy(y_pred[test_ind], labels[test_ind])

    print()
    print(f"Testing Accuracy loss: {test_loss:.3f}  |  Testing Accuracy: {test_acc:.2f}")
    return y_pred
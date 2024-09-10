import torch
import torch.nn as nn
import numpy as np
import time
from utils import * 

def train(model, features, labels, adj, train_set_ind, val_set_ind, config):

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    validation_acc = []
    validation_loss = []

    if config.use_early_stopping:
        last_min_val_loss = float('inf')
        patience_counter = 0
        stopped_early = False

    t_start = time.time()
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

            validation_loss.append(val_loss.item())
            validation_acc.append(val_acc)

            if config.use_early_stopping:
                if val_loss < last_min_val_loss:
                    last_min_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter == config.patience:
                        stopped_early = True
                        t_end = time.time()

        print(" | ".join([f"Epoch: {epoch:4d}", f"Train loss: {train_loss.item():.3f}",
                            f"Train acc: {train_acc:.2f}",
                            f"Val loss: {val_loss.item():.3f}",
                            f"Val acc: {val_acc:.2f}"]))

        if config.use_early_stopping and stopped_early:
            break

    if config.use_early_stopping and stopped_early:
        print(f"EARLY STOPPING condition met. Stopped at epoch: {epoch}.")
    else:
        t_end = time.time()

    print(f"Total training time: {t_end-t_start:.2f} seconds")

    return validation_acc, validation_loss


def evaluate_on_test(model, features, labels, adj, test_ind, config):

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()
        y_pred = model(features, adj)
        test_loss = criterion(y_pred[test_ind], labels[test_ind])
        test_acc = accuracy(y_pred[test_ind], labels[test_ind])

    print()
    print(f"Test loss: {test_loss:.3f}  |  Test acc: {test_acc:.2f}")
    return y_pred
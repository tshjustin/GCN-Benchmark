import torch
import torch.nn as nn
import torch.optim as optim

def setup_optimization(model, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

def train(model, optimizer, criterion, graph, features, labels, train_mask):
    model.train()
    logits = model(graph, features)
    logits = logits[train_mask]  # Apply mask to select training nodes
    loss = criterion(logits, labels[train_mask])  # Use masked labels for training
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    _, predicted = torch.max(logits, dim=1)
    correct = (predicted == labels[train_mask]).sum().item()
    accuracy = correct / len(labels[train_mask])

    return loss.item(), accuracy

def evaluate(model, mask, graph, features, labels, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]  # Apply mask to select validation/test nodes
        _, predicted = torch.max(logits, dim=1)
        correct = (predicted == labels[mask]).sum().item()
        accuracy = correct / len(labels[mask])
    
    loss = None
    if criterion is not None:
        loss = criterion(logits, labels[mask]).item()

    return accuracy, loss

import torch
import torch.nn as nn
import torch.optim as optim

def setup_optimization(model, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

def train(model, optimizer, criterion, graph, features, labels, train_mask, device='cpu'):
    model.train()
    graph = graph.to(device)  # Move graph to device
    features = features.to(device)  # Move features to device
    labels = labels.to(device)  # Move labels to device
    train_mask = train_mask.to(device)  # Move train_mask to device
    
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

def evaluate(model, mask, graph, features, labels, criterion, device='cpu'):
    model.eval()
    graph = graph.to(device)  # Move graph to device
    features = features.to(device)  # Move features to device
    labels = labels.to(device)  # Move labels to device
    mask = mask.to(device)  # Move mask to device
    
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

def train_loop(model, optimizer, criterion, graph, features, labels, train_mask, val_mask, epochs, device='cpu'):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, optimizer, criterion, graph, features, labels, train_mask, device)
        val_acc, val_loss = evaluate(model, val_mask, graph, features, labels, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if (epoch+1)%25==0:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies
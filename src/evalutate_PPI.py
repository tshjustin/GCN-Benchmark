import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from sklearn.metrics import f1_score

def collate_fn(batch):
    graphs, features, labels = zip(*[(g, g.ndata['feat'], g.ndata['label']) for g in batch])
    batched_graph = dgl.batch(graphs)
    batched_features = torch.cat(features)
    batched_labels = torch.cat(labels)
    return batched_graph, batched_features, batched_labels

def setup_optimization_ppi(model, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  
    return optimizer, criterion

def train(model, optimizer, criterion, batched_graph, batched_features, batched_labels, device):
    model.train()
    batched_graph = batched_graph.to(device)
    batched_features = batched_features.to(device) 
    batched_labels = batched_labels.to(device)
    
    logits = model(batched_graph, batched_features)  
    loss = criterion(logits, batched_labels)  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    predicted = (torch.sigmoid(logits) > 0.5).float()
    correct = (predicted == batched_labels).sum().item()
    accuracy = correct / batched_labels.numel()

    return loss.item(), accuracy

def evaluate_ppi(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for batched_graph, batched_features, batched_labels in data_loader:
            batched_graph = batched_graph.to(device)
            batched_features = batched_features.to(device)
            batched_labels = batched_labels.to(device)
            
            logits = model(batched_graph, batched_features)
            loss = criterion(logits, batched_labels)

            predicted = (torch.sigmoid(logits) > 0.5).float()
            correct = (predicted == batched_labels).sum().item()
            total_correct += correct
            total_samples += batched_labels.numel()
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples

    return accuracy, avg_loss

def train_loop_ppi(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    """Train Loop uses Batching instead"""
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        total_train_loss, total_train_acc = 0, 0
        model.train()
        for batched_graph, batched_features, batched_labels in train_loader:
            train_loss, train_acc = train(model, optimizer, criterion, batched_graph, batched_features, batched_labels, device)
            total_train_loss += train_loss
            total_train_acc += train_acc

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)

        val_acc, val_loss = evaluate_ppi(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if (epoch+1)%25==0:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies

# Using F1 Metrics 
def train_f1(model, optimizer, criterion, batched_graph, batched_features, batched_labels, device):
    """Using F1 Metrics instead of Accuracy"""
    model.train()
    batched_graph = batched_graph.to(device)
    batched_features = batched_features.to(device)
    batched_labels = batched_labels.to(device)
    
    logits = model(batched_graph, batched_features)
    loss = criterion(logits, batched_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    predicted = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
    labels = batched_labels.cpu().numpy()

    # Calculate F1 score (micro)
    f1 = f1_score(labels, predicted, average='micro')

    return loss.item(), f1

def evaluate_ppi_f1(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_f1 = 0, 0

    with torch.no_grad():
        for batched_graph, batched_features, batched_labels in data_loader:
            batched_graph = batched_graph.to(device)
            batched_features = batched_features.to(device) 
            batched_labels = batched_labels.to(device)
            
            logits = model(batched_graph, batched_features)
            loss = criterion(logits, batched_labels)

            predicted = (torch.sigmoid(logits) > 0.5).float().cpu().numpy() 
            labels = batched_labels.cpu().numpy() 

            # Calculate F1 score (micro)
            f1 = f1_score(labels, predicted, average='micro')

            total_f1 += f1
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    avg_f1 = total_f1 / len(data_loader)

    return avg_f1, avg_loss

def train_loop_ppi_f1(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    """Train Loop uses Batching and tracks F1 score instead of accuracy"""
    train_losses, train_f1_scores = [], []  
    val_losses, val_f1_scores = [], []  

    for epoch in range(epochs):
        total_train_loss, total_train_f1 = 0, 0 
        model.train()
        for batched_graph, batched_features, batched_labels in train_loader:
            train_loss, train_f1 = train(model, optimizer, criterion, batched_graph, batched_features, batched_labels, device)
            total_train_loss += train_loss
            total_train_f1 += train_f1

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_f1 = total_train_f1 / len(train_loader)
        train_losses.append(avg_train_loss)
        train_f1_scores.append(avg_train_f1)

        val_f1, val_loss = evaluate_ppi_f1(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train F1: {avg_train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    return train_losses, train_f1_scores, val_losses, val_f1_scores
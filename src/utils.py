from sklearn.metrics import f1_score
import torch

def accuracy(output, labels):
    """Calculate accuracy of model"""
    y_pred = output.max(1)[1].type_as(labels)
    correct = y_pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1(predictions, labels):
    """
    Modified F1 for multi-label classification. Each node can have multiple classes, thus for a node, 
    there are multipel labels each with their own probability score. 

    Instead of taking argmax (1 Output), take a threshold 
    """
    predictions = (torch.sigmoid(predictions) > 0.5).float() 
    
    predictions = predictions.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    f1 = f1_score(labels, predictions, average='micro')
    return f1

def accuracy(output, labels):
    """Calculate accuracy of model"""
    y_pred = output.max(1)[1].type_as(labels)
    correct = y_pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
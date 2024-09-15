import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

color_map = ["red", "blue", "green", "orange", "purple", "yellow", "brown"]

def visualize_embedding_tSNE(labels, out_features, num_classes):
    node_labels = labels.cpu().numpy()
    out_features = out_features.cpu().numpy()
    
    # Fit the t-SNE transformation
    t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(out_features)

    plt.figure()
    
    color_map = plt.get_cmap('tab10') 
    colors = [color_map(i) for i in range(num_classes)]

    for class_id in range(num_classes):
        plt.scatter(
            t_sne_embeddings[node_labels == class_id, 0],
            t_sne_embeddings[node_labels == class_id, 1], 
            s=20, 
            color=colors[class_id], 
            edgecolors='black', 
            linewidths=0.15, 
            label=f'Class {class_id}'
        )
    
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.axis("off")
    plt.title("t-SNE projection of the learned features")
    
    plt.show()

def visualize_performance(train_acc, train_loss, val_acc, val_loss, acc_color="blue", loss_color="red"):
    """Visualizes training and validation accuracy and loss across epochs."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Training Loss
    axs[0, 0].plot(train_loss, linewidth=2, color="green", label="Train Loss")
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_ylabel("Cross Entropy Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].grid()

    # Top-right: Training Accuracy
    axs[0, 1].plot(train_acc, linewidth=2, color="green", label="Train Accuracy")
    axs[0, 1].set_title("Training Accuracy")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].grid()

    # Bottom-left: Validation Loss
    axs[1, 0].plot(val_loss, linewidth=2, color=loss_color, label="Validation Loss")
    axs[1, 0].set_title("Validation Loss")
    axs[1, 0].set_ylabel("Cross Entropy Loss")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].grid()

    # Bottom-right: Validation Accuracy
    axs[1, 1].plot(val_acc, linewidth=2, color=acc_color, label="Validation Accuracy")
    axs[1, 1].set_title("Validation Accuracy")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embedding_tSNE(labels, out_features, num_classes):
    node_labels = labels.cpu().numpy()
    out_features = out_features.cpu().numpy()
    
    t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(out_features)

    plt.figure(figsize=(8, 6))
    
    color_map = plt.get_cmap('tab10') if num_classes <= 10 else plt.get_cmap('hsv')
    colors = [color_map(i / num_classes) for i in range(num_classes)]

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

def visualize_train_performance(results, acc_color="blue", loss_color="red"):
    """Visualizes training accuracy and loss across epochs for different configurations."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Iterate over results and plot
    for key, result in results.items():
        train_acc = result['train_accuracies']
        train_loss = result['train_losses']
        
        # Left: Training Loss
        axs[0].plot(train_loss, linewidth=2, label=f"{key} - Loss")
        
        # Right: Training Accuracy
        axs[1].plot(train_acc, linewidth=2, label=f"{key} - Accuracy")

    # Customize Loss Plot
    axs[0].set_title("Training Loss")
    axs[0].set_ylabel("Cross Entropy Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].grid()
    axs[0].legend()

    # Customize Accuracy Plot
    axs[1].set_title("Training Accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def visualize_val_performance(results, acc_color="blue", loss_color="red"):
    """Visualizes validation accuracy and loss across epochs for different configurations."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Iterate over results and plot
    for key, result in results.items():
        val_acc = result['val_accuracies']
        val_loss = result['val_losses']
        
        # Left: Validation Loss
        axs[0].plot(val_loss, linewidth=2, label=f"{key} - Loss")
        
        # Right: Validation Accuracy
        axs[1].plot(val_acc, linewidth=2, label=f"{key} - Accuracy")

    # Customize Loss Plot
    axs[0].set_title("Validation Loss")
    axs[0].set_ylabel("Cross Entropy Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].grid()
    axs[0].legend()

    # Customize Accuracy Plot
    axs[1].set_title("Validation Accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()
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

def visualize_train_performance(train_acc, train_loss, acc_color="blue", loss_color="green"):
    """Visualizes training accuracy and loss across epochs."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Training Loss
    axs[0].plot(train_loss, linewidth=2, color=loss_color, label="Train Loss")
    axs[0].set_title("Training Loss")
    axs[0].set_ylabel("Cross Entropy Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].grid()

    # Right: Training Accuracy
    axs[1].plot(train_acc, linewidth=2, color=acc_color, label="Train Accuracy")
    axs[1].set_title("Training Accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].grid()

    plt.tight_layout()
    plt.show()


def visualize_val_performance(val_acc, val_loss, acc_color="blue", loss_color="red"):
    """Visualizes validation accuracy and loss across epochs."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Validation Loss
    axs[0].plot(val_loss, linewidth=2, color=loss_color, label="Validation Loss")
    axs[0].set_title("Validation Loss")
    axs[0].set_ylabel("Cross Entropy Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].grid()

    # Right: Validation Accuracy
    axs[1].plot(val_acc, linewidth=2, color=acc_color, label="Validation Accuracy")
    axs[1].set_title("Validation Accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].grid()

    plt.tight_layout()
    plt.show()

def visualize_f1_performance(train_f1, val_f1, f1_color="purple"):
    """Visualizes training and validation F1 score across epochs."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Training F1 Score
    axs[0].plot(train_f1, linewidth=2, color=f1_color, label="Train Micro F1")
    axs[0].set_title("Training Micro F1")
    axs[0].set_ylabel("F1 Score")
    axs[0].set_xlabel("Epoch")
    axs[0].grid()

    # Right: Validation F1 Score
    axs[1].plot(val_f1, linewidth=2, color=f1_color, label="Validation Micro F1")
    axs[1].set_title("Validation Micro F1")
    axs[1].set_ylabel("F1 Score")
    axs[1].set_xlabel("Epoch")
    axs[1].grid()

    plt.tight_layout()
    plt.show()
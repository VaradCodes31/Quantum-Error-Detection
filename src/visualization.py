import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

def plot_training_history(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Train Loss", lw=2)
    plt.plot(epochs, val_loss, label="Val Loss", lw=2)
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="Train Acc", lw=2)
    plt.plot(epochs, val_acc, label="Val Acc", lw=2)
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/training_history.png", dpi=300)
    plt.close()

def plot_multiclass_roc(y_test, y_probs, classes):
    """Generates One-Vs-Rest ROC curves for each noise class."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    
    # Binarize labels for multiclass ROC
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    
    colors = sns.color_palette("husl", len(classes))
    
    for i, (color, class_name) in enumerate(zip(colors, classes)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC {class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (Multi-Class)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=300)
    plt.close()

def plot_multiclass_pr(y_test, y_probs, classes):
    """Generates Precision-Recall curves for each noise class."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    colors = sns.color_palette("husl", len(classes))
    
    for i, (color, class_name) in enumerate(zip(colors, classes)):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_probs[:, i])
        avg_precision = average_precision_score(y_test_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'PR {class_name} (AP = {avg_precision:.2f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (Multi-Class)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig("results/pr_curve.png", dpi=300)
    plt.close()

def plot_tsne_clusters(X, y, classes, n_samples=2000):
    """Visualizes feature space using t-SNE dimensionality reduction."""
    print(f"🚀 Running t-SNE reduction on {n_samples} samples...")
    
    # Subsample for speed
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(X_sub)

    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 10))
    
    scatter = sns.scatterplot(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1], 
        hue=[classes[i] for i in y_sub],
        palette="viridis",
        alpha=0.7,
        edgecolor='w',
        s=60
    )
    
    plt.title('t-SNE Clusters: Quantum Noise Feature Space', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title="Noise Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/tsne_plot.png", dpi=300)
    plt.close()
    print("✅ t-SNE plot saved to results/tsne_plot.png")
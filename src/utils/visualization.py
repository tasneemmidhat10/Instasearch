import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def plot_spectrum(mz, intensity, title="Mass Spectrum"):
    plt.figure(figsize=(10, 4))
    plt.stem(mz, intensity, markerfmt=" ", basefmt="b-")
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_metrics(history):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['loss'], color=color, marker='o', label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], color=color, marker='x', linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(history['acc'], color=color, marker='s', label='Train Accuracy')
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], color=color, marker='d', linestyle='--', label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title("Training & Validation Metrics")
    fig.tight_layout()
    plt.show()

def plot_similarity_matrix(z_spec, z_pep):
    # Compute cosine similarity matrix
    sim = (z_spec @ z_pep.T).cpu().numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(sim, annot=False, cmap='viridis')
    plt.title("Spectrum-Peptide Similarity Matrix (Batch)")
    plt.xlabel("Peptide Index")
    plt.ylabel("Spectrum Index")
    plt.show()

def plot_embeddings(z_spec, z_pep, title="Embedding Visualization (t-SNE)"):
    # Concatenate embeddings
    z_s = z_spec.cpu().numpy()
    z_p = z_pep.cpu().numpy()
    z_all = np.concatenate([z_s, z_p], axis=0)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z_all)

    # Split back
    z_s_2d = z_2d[:len(z_s)]
    z_p_2d = z_2d[len(z_s):]

    plt.figure(figsize=(10, 8))
    plt.scatter(z_s_2d[:, 0], z_s_2d[:, 1], alpha=0.6, label='Spectra', c='tab:red', marker='o')
    plt.scatter(z_p_2d[:, 0], z_p_2d[:, 1], alpha=0.6, label='Peptides', c='tab:blue', marker='x')

    # Draw lines between matching pairs
    for i in range(len(z_s_2d)):
        plt.plot([z_s_2d[i, 0], z_p_2d[i, 0]], [z_s_2d[i, 1], z_p_2d[i, 1]], 'k-', alpha=0.1)

    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.3)
    plt.show()
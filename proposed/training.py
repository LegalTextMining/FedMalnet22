"""
train.py

Fungsi-fungsi training, evaluasi, dan analisis fitur laten (PCA progres)
untuk model FFNN (theta + phi) di model.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from model import build_model


# =============================================================================
# Training (mini-batch) + capture fitur laten theta(x) tiap N epoch
# Cukup dipanggil SEKALI: dapat model terlatih + snapshot progres PCA sekaligus
# =============================================================================
def train_and_track(split, in_dim, num_classes=2, epochs=100, lr=1e-3,
                     batch_size=64, capture_every=10, seed=42, device=None):
    """
    Training mini-batch sambil menyimpan snapshot fitur laten theta(x)
    dari test set setiap `capture_every` epoch. Hanya training SATU KALI,
    tidak perlu dipanggil dua fungsi terpisah lagi.

    Returns:
        model, feature_snapshots (dict {epoch: tensor}),
        weights_before (dict), weights_after (dict)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = split["X_train_t"].float()
    y_train_t = split["y_train_t"].long()
    X_val_t   = split["X_val_t"].float().to(device)
    y_val_t   = split["y_val_t"].long().to(device)
    X_test_t  = split["X_test_t"].float().to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = build_model(in_dim=in_dim, num_classes=num_classes, seed=seed).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # bobot theta SEBELUM training
    weights_before = {
        name: param.clone().detach().cpu()
        for name, param in model.theta.named_parameters()
    }

    feature_snapshots = {}

    # snapshot awal (epoch 0, sebelum training)
    model.eval()
    with torch.no_grad():
        feature_snapshots[0] = model.theta(X_test_t).clone().cpu()
        features_before = feature_snapshots[0]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_ds)
        do_capture = (epoch + 1) % capture_every == 0 or (epoch + 1) == epochs

        if do_capture:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t)
                val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean()

                feature_snapshots[epoch + 1] = model.theta(X_test_t).clone().cpu()

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} "
                  f"- Val Loss: {val_loss.item():.4f} - Val Acc: {val_acc.item()*100:.2f}%")

    print(f"\nJumlah snapshot tersimpan: {len(feature_snapshots)}")
    print("Epoch yang di-capture:", list(feature_snapshots.keys()))

    # bobot & fitur SESUDAH training
    weights_after = {
        name: param.clone().detach().cpu()
        for name, param in model.theta.named_parameters()
    }
    model.eval()
    with torch.no_grad():
        features_after = model.theta(X_test_t).clone().cpu()

    print("\n=== Fitur SESUDAH training (5 sampel pertama) ===")
    print(features_after[:5])

    diff = (features_after - features_before).abs().mean()
    print(f"\nRata-rata perubahan nilai fitur laten (before vs after): {diff.item():.4f}")

    print("\n=== Perubahan bobot (before vs after) ===")
    for name in weights_before:
        w_diff = (weights_after[name] - weights_before[name]).abs()
        print(f"{name}: rata-rata perubahan={w_diff.mean():.4f}, "
              f"perubahan maksimum={w_diff.max():.4f}")

    return model, feature_snapshots, weights_before, weights_after


# =============================================================================
# Evaluasi di test set (pakai model yang sama, tidak training ulang)
# =============================================================================
def evaluate_model(model, split, target_names=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test_t = split["X_test_t"].float().to(device)
    y_test_t = split["y_test_t"].long()

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        preds = logits.argmax(dim=1).cpu()

    print(classification_report(y_test_t, preds, target_names=target_names))
    print("Confusion Matrix:\n", confusion_matrix(y_test_t, preds))

    return preds


# =============================================================================
# Visualisasi PCA: fitur laten before vs after
# =============================================================================
def plot_pca_before_after(feature_snapshots, y_test, target_names,
                           save_path="feature_before_after.png"):
    epoch_keys = list(feature_snapshots.keys())
    first_ep, last_ep = epoch_keys[0], epoch_keys[-1]

    features_before = feature_snapshots[first_ep].numpy()
    features_after  = feature_snapshots[last_ep].numpy()

    pca_before = PCA(n_components=2).fit_transform(features_before)
    pca_after  = PCA(n_components=2).fit_transform(features_after)

    colors = ["tab:blue", "tab:red"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    y_test = np.asarray(y_test)
    for cls in range(2):
        idx = (y_test == cls)
        axes[0].scatter(pca_before[idx, 0], pca_before[idx, 1],
                         label=target_names[cls], alpha=0.6, s=15, c=colors[cls])
        axes[1].scatter(pca_after[idx, 0], pca_after[idx, 1],
                         label=target_names[cls], alpha=0.6, s=15, c=colors[cls])

    axes[0].set_title(f"Fitur θ(x) SEBELUM training\n(epoch {first_ep})")
    axes[1].set_title(f"Fitur θ(x) SESUDAH training\n(epoch {last_ep})")
    for ax in axes:
        ax.legend()
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()


# =============================================================================
# Visualisasi PCA per epoch (progres representasi laten)
# =============================================================================
def plot_pca_progress(feature_snapshots, y_test, target_names,
                       save_path="pca_progress_per_epoch.png"):
    epoch_list = list(feature_snapshots.keys())
    n_snapshots = len(epoch_list)

    n_cols = 4
    n_rows = int(np.ceil(n_snapshots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    colors = ["tab:blue", "tab:red"]
    y_test = np.asarray(y_test)

    for i, ep in enumerate(epoch_list):
        feats = feature_snapshots[ep].numpy()
        pca_result = PCA(n_components=2).fit_transform(feats)

        ax = axes[i]
        for cls in range(2):
            idx = (y_test == cls)
            ax.scatter(pca_result[idx, 0], pca_result[idx, 1],
                       label=target_names[cls], alpha=0.6, s=10, c=colors[cls])

        title = "Epoch 0 (awal)" if ep == 0 else f"Epoch {ep}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)

        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n_snapshots, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Progres Fitur Laten θ(x) via PCA — per Epoch", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
"""
train.py
========
Menghubungkan dataset.py (split/scale/tensor) dengan model.py (FFNNModel)
untuk training, validasi, dan evaluasi.

Contoh pemakaian dasar:
------------------------
from dataset import prepare_dataset
from model import build_model
from train import train_model, evaluate

data = prepare_dataset(X, y, task="classification")
model = build_model(in_dim=X.shape[1], num_classes=2)

history = train_model(model, data, epochs=50, lr=1e-3)
test_acc = evaluate(model, data["test"])
"""

import torch
import torch.nn as nn
import copy


def train_model(model, data, epochs=50, lr=1e-3, batch_size=32,
                 weight_decay=0.0, patience=10, device=None, verbose=True):
    """
    Melatih model menggunakan train set, memantau val set, dengan early stopping.

    Parameters
    ----------
    model : nn.Module
        Model yang akan dilatih (mis. FFNNModel dari model.py).
    data : dict
        Output dari `prepare_dataset()` di dataset.py, minimal berisi
        data["train"] = (X_train_t, y_train_t) dan data["val"] = (X_val_t, y_val_t).
    epochs : int
        Jumlah maksimum epoch.
    lr : float
        Learning rate untuk optimizer Adam.
    batch_size : int
        Ukuran mini-batch.
    weight_decay : float
        L2 regularization di optimizer.
    patience : int
        Berhenti lebih awal jika val loss tidak membaik setelah `patience` epoch.
    device : str, optional
        "cuda" atau "cpu". Default: otomatis deteksi.
    verbose : bool
        Cetak log tiap epoch.

    Returns
    -------
    history : dict
        {"train_loss": [...], "val_loss": [...], "val_acc": [...]}
        Model di dalam `model` sudah otomatis di-restore ke bobot terbaik (val loss terendah).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    n_samples = X_train.shape[0]
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        train_loss = epoch_loss / n_samples

        # Validasi
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping di epoch {epoch} (val_loss tidak membaik selama {patience} epoch).")
                break

    model.load_state_dict(best_state)
    return history


def evaluate(model, xy, device=None):
    """
    Evaluasi model pada suatu set (val/test), mengembalikan accuracy.

    Parameters
    ----------
    model : nn.Module
    xy : tuple (X_t, y_t)
        Mis. data["test"] dari prepare_dataset().
    device : str, optional

    Returns
    -------
    acc : float
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X, y = xy
    X, y = X.to(device), y.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()

    return acc


if __name__ == "__main__":
    # Contoh end-to-end dengan dataset dummy
    from sklearn.datasets import load_breast_cancer
    from dataset import prepare_dataset
    from model import build_model

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    data = prepare_dataset(X, y, val_size=0.15, test_size=0.15, task="classification")

    in_dim = data["train"][0].shape[1]
    num_classes = int(y.nunique())

    model = build_model(in_dim=in_dim, num_classes=num_classes)

    history = train_model(model, data, epochs=50, lr=1e-3, batch_size=32, patience=10)
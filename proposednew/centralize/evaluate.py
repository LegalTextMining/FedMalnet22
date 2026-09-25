"""
evaluate.py
===========
Evaluasi model dengan metrik: accuracy, precision, recall, F1-score
(+ confusion matrix & classification report).

Contoh pemakaian:
------------------
from evaluate import evaluate_model

metrics = evaluate_model(model, data["test"], average="macro")
print(metrics)
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def get_predictions(model, xy, device=None):
    """
    Ambil prediksi model pada suatu set (val/test).

    Parameters
    ----------
    model : nn.Module
    xy : tuple (X_t, y_t)
    device : str, optional

    Returns
    -------
    y_true, y_pred : numpy array
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X, y = xy
    X, y = X.to(device), y.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)

    y_true = y.cpu().numpy()
    y_pred = preds.cpu().numpy()
    return y_true, y_pred


def evaluate_model(model, xy, class_names=None, device=None, verbose=True):
    """
    Evaluasi lengkap: accuracy, precision, recall, F1-score, confusion matrix.

    Precision, recall, dan F1-score dihitung dengan average="macro"
    (rata-rata sederhana antar kelas) secara tetap — tanpa opsi
    micro/weighted.

    Parameters
    ----------
    model : nn.Module
    xy : tuple (X_t, y_t)
        Mis. data["test"] atau data["val"] dari prepare_dataset().
    class_names : list of str, optional
        Nama kelas untuk classification_report yang lebih terbaca.
    device : str, optional
    verbose : bool
        Jika True, cetak classification_report & confusion matrix.

    Returns
    -------
    metrics : dict
        {
          "accuracy": float,
          "precision": float,
          "recall": float,
          "f1": float,
          "confusion_matrix": np.ndarray,
          "report": str  # classification_report versi teks
        }
    """
    y_true, y_pred = get_predictions(model, xy, device=device)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }

    if verbose:
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

    return metrics


if __name__ == "__main__":
    # Contoh end-to-end
    from sklearn.datasets import load_breast_cancer
    from dataset import prepare_dataset
    from model import build_model
    from training import train_model

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    data = prepare_dataset(X, y, task="classification")

    in_dim = data["train"][0].shape[1]
    num_classes = int(y.nunique())
    model = build_model(in_dim=in_dim, num_classes=num_classes)

    train_model(model, data, epochs=30, verbose=False)
"""
dataset.py
==========
Modul reusable untuk melakukan split (train/val/test), scaling,
dan konversi ke tensor PyTorch. Tinggal panggil `prepare_dataset(X, y)`
untuk dataset apapun.

Contoh pemakaian:
------------------
from dataset import prepare_dataset

data = prepare_dataset(X, y)

X_train_t, y_train_t = data["train"]
X_val_t, y_val_t     = data["val"]
X_test_t, y_test_t   = data["test"]
scaler                = data["scaler"]
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_dataset(X, y, val_size=0.15, test_size=0.15, random_state=42, stratify=True):
    """
    Split dataset menjadi train, validation, dan test set.

    Parameters
    ----------
    X : array-like / DataFrame
        Fitur.
    y : array-like / Series
        Target/label.
    val_size : float
        Proporsi data untuk validation set (dari total dataset).
    test_size : float
        Proporsi data untuk test set (dari total dataset).
    random_state : int
        Seed untuk reproducibility.
    stratify : bool
        Jika True, split akan mempertahankan proporsi kelas (cocok untuk klasifikasi).

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    temp_size = val_size + test_size
    strat_full = y if stratify else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        random_state=random_state,
        stratify=strat_full
    )

    # Proporsi test terhadap temp (val+test)
    relative_test_size = test_size / temp_size
    strat_temp = y_temp if stratify else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=strat_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, scaler=None):
    """
    Scaling fitur menggunakan StandardScaler (fit hanya di train).

    Parameters
    ----------
    scaler : sklearn scaler instance, optional
        Jika ingin pakai scaler lain (MinMaxScaler, dll). Default: StandardScaler.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    if scaler is None:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def to_tensors(X_train, X_val, X_test, y_train, y_val, y_test,
               task="classification"):
    """
    Konversi array hasil scaling + label ke tensor PyTorch.

    Parameters
    ----------
    task : str
        "classification" -> y bertipe long (int label)
        "regression"      -> y bertipe float32

    Returns
    -------
    dict berisi tensor train/val/test
    """
    def to_X_tensor(X):
        if hasattr(X, "values"):
            X = X.values
        return torch.tensor(np.asarray(X), dtype=torch.float32)

    def to_y_tensor(y):
        if hasattr(y, "values"):
            y = y.values
        y = np.asarray(y)
        dtype = torch.long if task == "classification" else torch.float32
        return torch.tensor(y, dtype=dtype)

    X_train_t = to_X_tensor(X_train)
    X_val_t = to_X_tensor(X_val)
    X_test_t = to_X_tensor(X_test)

    y_train_t = to_y_tensor(y_train)
    y_val_t = to_y_tensor(y_val)
    y_test_t = to_y_tensor(y_test)

    return {
        "train": (X_train_t, y_train_t),
        "val": (X_val_t, y_val_t),
        "test": (X_test_t, y_test_t),
    }


def prepare_dataset(X, y, val_size=0.15, test_size=0.15, random_state=42,
                     stratify=True, task="classification", scaler=None, verbose=True):
    """
    Pipeline lengkap: split -> scale -> tensor.
    Tinggal panggil fungsi ini untuk dataset apapun.

    Parameters
    ----------
    X, y : dataset fitur & target (array-like / DataFrame / Series)
    val_size, test_size : proporsi val & test dari total dataset
    random_state : seed reproducibility
    stratify : pertahankan proporsi kelas saat split (klasifikasi)
    task : "classification" atau "regression" (menentukan dtype tensor y)
    scaler : instance scaler custom (opsional), default StandardScaler
    verbose : cetak shape tiap split

    Returns
    -------
    dict:
        {
          "train": (X_train_t, y_train_t),
          "val":   (X_val_t, y_val_t),
          "test":  (X_test_t, y_test_t),
          "scaler": fitted scaler,
          "raw": (X_train, X_val, X_test, y_train, y_val, y_test)  # sebelum scaling
        }
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, val_size=val_size, test_size=test_size,
        random_state=random_state, stratify=stratify
    )

    X_train_s, X_val_s, X_test_s, fitted_scaler = scale_features(
        X_train, X_val, X_test, scaler=scaler
    )

    tensors = to_tensors(
        X_train_s, X_val_s, X_test_s,
        y_train, y_val, y_test,
        task=task
    )

    if verbose:
        print("Train :", tensors["train"][0].shape, tensors["train"][1].shape)
        print("Val   :", tensors["val"][0].shape, tensors["val"][1].shape)
        print("Test  :", tensors["test"][0].shape, tensors["test"][1].shape)

    tensors["scaler"] = fitted_scaler
    tensors["raw"] = (X_train, X_val, X_test, y_train, y_val, y_test)

    return tensors


if __name__ == "__main__":
    # Contoh pemakaian cepat dengan dataset dummy
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    result = prepare_dataset(X, y, val_size=0.15, test_size=0.15, task="classification")
    X_train_t, y_train_t = result["train"]
    print(X_train_t[:2])
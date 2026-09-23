"""
dataset.py

Split data (train/val/test), scaling, konversi tensor, dan partisi client (IID / Non-IID).
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_prepare_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio,
        random_state=random_state, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    f = lambda a: torch.tensor(a, dtype=torch.float32)
    l = lambda a: torch.tensor(a, dtype=torch.long)

    print("Train:", X_train.shape, "| Val:", X_val.shape, "| Test:", X_test.shape)

    return {
        "X_train_t": f(X_train), "X_val_t": f(X_val), "X_test_t": f(X_test),
        "y_train_t": l(y_train), "y_val_t": l(y_val), "y_test_t": l(y_test),
        "scaler": scaler,
    }


def partition_clients(y, n_clients=4, mode="iid", alpha=0.5, seed=42, min_size=10):
    """Return list of index array per client. mode: 'iid' | 'non_iid' (Dirichlet)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    if mode == "iid":
        return np.array_split(rng.permutation(len(y)), n_clients)
    if mode != "non_iid":
        raise ValueError("mode harus 'iid' atau 'non_iid'")

    while True:  # ulangi sampai tiap client punya >= min_size sampel
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            cuts = (np.cumsum(rng.dirichlet([alpha] * n_clients)) * len(idx)).astype(int)[:-1]
            for k, p in enumerate(np.split(idx, cuts)):
                parts[k].extend(p)
        if min(map(len, parts)) >= min_size:
            return [np.array(p) for p in parts]


def make_client_datasets(split, **kw):
    """Bagi train set jadi list client {'X','y'}. kw diteruskan ke partition_clients."""
    y = split["y_train_t"].numpy()
    return [
        {"X": split["X_train_t"][i], "y": split["y_train_t"][i]}
        for i in map(torch.as_tensor, partition_clients(y, **kw))
    ]


def print_client_distribution(clients, title=""):
    print(f"== {title} ==")
    for i, c in enumerate(clients):
        print(f"  Client {i}: n={len(c['y'])} | kelas={torch.bincount(c['y'], minlength=2).tolist()}")

def make_federated_splits(X, y, n_clients=4, mode="iid", alpha=0.5,
                          test_size=0.15, val_size=0.15, seed=42):
    """
    Bagi data ke client dulu (IID / Non-IID), lalu tiap client di-split
    train/val/test sendiri dengan stratifikasi dan scaler sendiri.

    Return: list of dict per client, isi:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    y = np.asarray(y)
    X = np.asarray(X)
    vr = val_size / (1 - test_size)
    clients = []

    for idx in partition_clients(y, n_clients, mode, alpha, seed):
        Xc, yc = X[idx], y[idx]
        Xtv, Xte, ytv, yte = train_test_split(
            Xc, yc, test_size=test_size, random_state=seed, stratify=yc)
        Xtr, Xva, ytr, yva = train_test_split(
            Xtv, ytv, test_size=vr, random_state=seed, stratify=ytv)

        sc = StandardScaler().fit(Xtr)
        f = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32)
        l = lambda a: torch.tensor(a, dtype=torch.long)

        clients.append({
            "X_train": f(Xtr), "y_train": l(ytr),
            "X_val":   f(Xva), "y_val":   l(yva),
            "X_test":  f(Xte), "y_test":  l(yte),
            "scaler": sc,
        })
    return clients


def print_federated_splits(clients, title=""):
    print(f"== {title} ==")
    for i, c in enumerate(clients):
        cnt = lambda k: torch.bincount(c[k], minlength=2).tolist()
        print(f"  Client {i}: train={len(c['y_train'])}{cnt('y_train')} "
              f"| val={len(c['y_val'])}{cnt('y_val')} "
              f"| test={len(c['y_test'])}{cnt('y_test')}")
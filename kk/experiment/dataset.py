"""
dataset.py

Split data (train/val/test), oversampling pada data training,
scaling, konversi tensor, dan partisi client (IID / Non-IID).
"""

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import (
    SMOTE,
    BorderlineSMOTE,
    ADASYN
)

from imblearn.combine import SMOTETomek


# ============================================================
# 1. OVERSAMPLING
# ============================================================

def apply_oversampling(
    X,
    y,
    method=None,
    seed=42
):
    """
    Melakukan oversampling pada data training.

    Parameters
    ----------
    X : array-like
        Fitur training.

    y : array-like
        Label training.

    method : str or None
        Pilihan:
        - None
        - "random"
        - "smote"
        - "borderline_smote"
        - "smote_tomek"

    seed : int
        Random seed.

    Returns
    -------
    X_resampled, y_resampled
    """

    if method is None:
        return X, y

    if method == "random":

        sampler = ADASYN(
            random_state=seed
        )

    elif method == "smote":

        sampler = SMOTE(
            random_state=seed
        )

    elif method == "borderline_smote":

        sampler = BorderlineSMOTE(
            random_state=seed
        )

    elif method == "smote_tomek":

        sampler = SMOTETomek(
            random_state=seed
        )

    else:
        raise ValueError(
            "method harus salah satu dari: "
            "None, 'random', 'smote', "
            "'borderline_smote', 'smote_tomek'"
        )

    X_resampled, y_resampled = sampler.fit_resample(
        X,
        y
    )

    return X_resampled, y_resampled


# ============================================================
# 2. SPLIT DATA GLOBAL
# ============================================================

def split_and_prepare_data(
    X,
    y,
    test_size=0.15,
    val_size=0.15,
    random_state=42
):
    """
    Split data menjadi train/validation/test.

    Oversampling tidak dilakukan di sini.
    """

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val
    )

    # --------------------------------------------------------
    # StandardScaler hanya berdasarkan TRAIN
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # --------------------------------------------------------
    # Konversi tensor
    # --------------------------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    print(
        "Train:", X_train.shape,
        "| Val:", X_val.shape,
        "| Test:", X_test.shape
    )

    return {
        "X_train_t": X_train,
        "X_val_t": X_val,
        "X_test_t": X_test,

        "y_train_t": y_train,
        "y_val_t": y_val,
        "y_test_t": y_test,

        "scaler": scaler,
    }


# ============================================================
# 3. PARTITION CLIENT
# ============================================================

def partition_clients(
    y,
    n_clients=4,
    mode="iid",
    alpha=0.5,
    seed=42,
    min_size=10
):
    """
    Membagi indeks data ke beberapa client.

    mode:
        - "iid"
        - "non_iid"

    Non-IID menggunakan distribusi Dirichlet.
    """

    rng = np.random.default_rng(seed)

    y = np.asarray(y)

    # --------------------------------------------------------
    # IID
    # --------------------------------------------------------

    if mode == "iid":

        indices = rng.permutation(len(y))

        return np.array_split(
            indices,
            n_clients
        )

    # --------------------------------------------------------
    # Validasi mode
    # --------------------------------------------------------

    if mode != "non_iid":
        raise ValueError(
            "mode harus 'iid' atau 'non_iid'"
        )

    # --------------------------------------------------------
    # Non-IID Dirichlet
    # --------------------------------------------------------

    while True:

        parts = [
            []
            for _ in range(n_clients)
        ]

        for c in np.unique(y):

            class_indices = np.where(
                y == c
            )[0]

            class_indices = rng.permutation(
                class_indices
            )

            proportions = rng.dirichlet(
                [alpha] * n_clients
            )

            cuts = (
                np.cumsum(proportions)
                * len(class_indices)
            ).astype(int)[:-1]

            split_indices = np.split(
                class_indices,
                cuts
            )

            for client_id, p in enumerate(
                split_indices
            ):

                parts[client_id].extend(
                    p.tolist()
                )

        # ----------------------------------------------------
        # Pastikan setiap client memiliki minimal data
        # ----------------------------------------------------

        if min(map(len, parts)) >= min_size:

            return [
                np.array(p)
                for p in parts
            ]


# ============================================================
# 4. MEMBUAT DATASET CLIENT
# ============================================================

def make_client_datasets(
    split,
    **kwargs
):
    """
    Membagi train global menjadi dataset client.

    Fungsi ini menggunakan data yang sudah di-scaling.
    """

    y = split["y_train_t"].numpy()

    client_indices = partition_clients(
        y,
        **kwargs
    )

    clients = []

    for indices in client_indices:

        indices = torch.as_tensor(
            indices,
            dtype=torch.long
        )

        clients.append({
            "X": split["X_train_t"][indices],
            "y": split["y_train_t"][indices]
        })

    return clients


# ============================================================
# 5. FEDERATED SPLITS
# ============================================================

def make_federated_splits(
    X,
    y,
    n_clients=4,
    mode="iid",
    alpha=0.5,
    test_size=0.15,
    val_size=0.15,
    seed=42,
    oversample_method=None
):
    """
    Membagi data menjadi beberapa client.

    Pipeline:

        Dataset
           ↓
        Partition Client
           ↓
        Split Train / Validation / Test
           ↓
        Oversampling TRAIN
           ↓
        StandardScaler
           ↓
        Tensor

    Oversampling hanya dilakukan pada TRAIN.

    Parameters
    ----------
    oversample_method : str or None

        None
            Tidak menggunakan oversampling.

        "random"
            Random Oversampling.

        "smote"
            SMOTE.

        "borderline_smote"
            Borderline-SMOTE.

        "smote_tomek"
            SMOTE-Tomek.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    vr = val_size / (1 - test_size)

    clients = []

    # ========================================================
    # PARTITION CLIENT
    # ========================================================

    client_indices = partition_clients(
        y=y,
        n_clients=n_clients,
        mode=mode,
        alpha=alpha,
        seed=seed
    )

    # ========================================================
    # PROSES SETIAP CLIENT
    # ========================================================

    for client_id, idx in enumerate(client_indices):

        Xc = X[idx]
        yc = y[idx]

        # ----------------------------------------------------
        # Split Train + Test
        # ----------------------------------------------------

        Xtv, Xte, ytv, yte = train_test_split(
            Xc,
            yc,
            test_size=test_size,
            random_state=seed,
            stratify=yc
        )

        # ----------------------------------------------------
        # Split Train + Validation
        # ----------------------------------------------------

        Xtr, Xva, ytr, yva = train_test_split(
            Xtv,
            ytv,
            test_size=vr,
            random_state=seed,
            stratify=ytv
        )

        # ----------------------------------------------------
        # DISTRIBUSI SEBELUM OVERSAMPLING
        # ----------------------------------------------------

        before_counts = np.bincount(
            ytr,
            minlength=2
        )

        # ----------------------------------------------------
        # OVERSAMPLING TRAIN SAJA
        # ----------------------------------------------------

        Xtr, ytr = apply_oversampling(
            Xtr,
            ytr,
            method=oversample_method,
            seed=seed + client_id
        )

        # ----------------------------------------------------
        # DISTRIBUSI SESUDAH OVERSAMPLING
        # ----------------------------------------------------

        after_counts = np.bincount(
            ytr,
            minlength=2
        )

        # ----------------------------------------------------
        # STANDARDISASI
        #
        # FIT HANYA PADA TRAIN SET
        # ----------------------------------------------------

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(
            Xtr
        )

        Xva = scaler.transform(
            Xva
        )

        Xte = scaler.transform(
            Xte
        )

        # ----------------------------------------------------
        # KONVERSI KE TENSOR
        # ----------------------------------------------------

        Xtr = torch.tensor(
            Xtr,
            dtype=torch.float32
        )

        ytr = torch.tensor(
            ytr,
            dtype=torch.long
        )

        Xva = torch.tensor(
            Xva,
            dtype=torch.float32
        )

        yva = torch.tensor(
            yva,
            dtype=torch.long
        )

        Xte = torch.tensor(
            Xte,
            dtype=torch.float32
        )

        yte = torch.tensor(
            yte,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # SIMPAN CLIENT
        # ----------------------------------------------------

        clients.append({

            "X_train": Xtr,
            "y_train": ytr,

            "X_val": Xva,
            "y_val": yva,

            "X_test": Xte,
            "y_test": yte,

            "scaler": scaler,

            "before_counts": before_counts,
            "after_counts": after_counts,

        })

    return clients


# ============================================================
# 6. PRINT DISTRIBUSI CLIENT
# ============================================================

def print_federated_splits(
    clients,
    title=""
):

    print(f"\n== {title} ==")

    for i, client in enumerate(clients):

        train_count = torch.bincount(
            client["y_train"],
            minlength=2
        ).tolist()

        val_count = torch.bincount(
            client["y_val"],
            minlength=2
        ).tolist()

        test_count = torch.bincount(
            client["y_test"],
            minlength=2
        ).tolist()

        print(
            f"Client {i}: "
            f"train={len(client['y_train'])} "
            f"{train_count} | "
            f"val={len(client['y_val'])} "
            f"{val_count} | "
            f"test={len(client['y_test'])} "
            f"{test_count}"
        )


# ============================================================
# 7. PRINT DISTRIBUSI SEBELUM DAN SESUDAH OVERSAMPLING
# ============================================================

def print_oversampling_distribution(
    clients,
    title=""
):

    print(f"\n== {title} ==")

    for i, client in enumerate(clients):

        before = client["before_counts"]
        after = client["after_counts"]

        print(
            f"Client {i}: "
            f"before={before.tolist()} "
            f"→ after={after.tolist()}"
        )
"""
dataset.py

Split data menjadi Train, Validation, Test, lalu melakukan scaling
dan konversi ke tensor PyTorch.
"""

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_prepare_data(
    X,
    y,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    print(
        "Train:", X_train_t.shape,
        "| Val:", X_val_t.shape,
        "| Test:", X_test_t.shape,
    )

    return {
        "X_train_t": X_train_t,
        "X_val_t": X_val_t,
        "X_test_t": X_test_t,
        "y_train_t": y_train_t,
        "y_val_t": y_val_t,
        "y_test_t": y_test_t,
        "scaler": scaler,
    }

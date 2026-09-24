"""
fed.py
Federated Learning pipeline with Flower (flwr) + PyTorch.

Assumes you already built `clients` via:

    clients = make_federated_splits(
        X=X, y=y,
        n_clients=4,
        mode="non_iid", alpha=0.5,
        test_size=0.15, val_size=0.15, seed=42,
        oversample_method="smote",
    )

`clients` is expected to be indexable by client id (list or dict with keys
"0", "1", ... or 0, 1, ...) where each entry is a dict-like object with:
    "X_train", "y_train", "X_val", "y_val", "X_test", "y_test"

If your `make_federated_splits` returns different key names (e.g. "x_train"
or a namedtuple), just edit `get_client_data()` below — everything else
stays the same.
"""

import os
import copy
import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, Context, NDArrays, Metrics, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from model import build_model


# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLIENTS = 4            # must match n_clients used in make_federated_splits
NUM_ROUNDS = 10
LOCAL_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0


# =========================================================
# DATA ACCESS HELPERS
# =========================================================
def get_client_data(clients, cid: int):
    """
    Pull one client's split out of the `clients` object returned by
    make_federated_splits(). Edit the key names here if yours differ.
    """
    # Support both list-like and dict-like (str or int keyed) containers
    if isinstance(clients, dict):
        entry = clients.get(cid, clients.get(str(cid)))
    else:
        entry = clients[cid]

    X_train, y_train = entry["X_train"], entry["y_train"]
    X_val, y_val = entry["X_val"], entry["y_val"]
    X_test, y_test = entry["X_test"], entry["y_test"]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y), dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# =========================================================
# MODEL PARAMETER <-> NDARRAYS HELPERS
# =========================================================
def compute_num_classes(y) -> int:
    """
    Number of classes = max label value + 1 (NOT len(set(y))!).
    len(set(y)) only counts how many *distinct* values appear, which can
    under-count if labels aren't contiguous from 0..n-1, or if you compute
    it on a subset. Using max()+1 guarantees the model's output layer is
    large enough for the highest label index that ever occurs.
    """
    return int(np.max(y)) + 1


def get_parameters(model: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# =========================================================
# LOCAL TRAIN / EVAL
# =========================================================
def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float) -> None:
    model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
    model.to(DEVICE)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(yb.cpu().numpy().tolist())

    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
    return avg_loss, metrics


# =========================================================
# FLOWER CLIENT
# =========================================================
class FlowerClient(NumPyClient):
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, test_loader: DataLoader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        lr = float(config.get("lr", LR))
        train(self.model, self.train_loader, epochs=epochs, lr=lr)
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, metrics = evaluate(self.model, self.val_loader)
        return float(loss), len(self.val_loader.dataset), metrics


def make_client_fn(clients, input_dim: int, num_classes: int, feature_dim: int = 64):
    def client_fn(context: Context) -> Client:
        cid = int(context.node_config["partition-id"])

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_client_data(clients, cid)

        train_loader = to_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
        val_loader = to_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
        test_loader = to_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

        model = build_model(input_dim=input_dim, feature_dim=feature_dim, n_classes=num_classes).to(DEVICE)

        return FlowerClient(model, train_loader, val_loader, test_loader).to_client()

    return client_fn


# =========================================================
# METRIC AGGREGATION (weighted by num examples)
# =========================================================
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(n for n, _ in metrics)
    agg = {}
    if total_examples == 0:
        return agg
    for key in metrics[0][1].keys():
        agg[key] = sum(n * m[key] for n, m in metrics) / total_examples
    return agg


# =========================================================
# SERVER
# =========================================================
class RecordingFedAvg(FedAvg):
    """
    FedAvg that stores per-round results in `self.history` as it goes.
    Needed because `run_simulation()` (Flower's ServerApp/ClientApp style)
    does NOT return a History object like the old start_simulation() did.
    Since ServerApp runs in-process (same Python process as your notebook,
    just a separate thread), appending to a plain list here works fine and
    is readable afterwards via `strategy.history`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: Dict[str, list] = {
            "losses_distributed": [],
            "metrics_distributed_fit": [],
            "metrics_distributed": [],
        }

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        self.history["metrics_distributed_fit"].append((server_round, aggregated_metrics))
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        self.history["losses_distributed"].append((server_round, aggregated_loss))
        self.history["metrics_distributed"].append((server_round, aggregated_metrics))
        print(f"[Round {server_round}] loss={aggregated_loss:.4f} | {aggregated_metrics}")
        return aggregated_loss, aggregated_metrics


def make_server_fn(input_dim: int, num_classes: int, num_rounds: int, feature_dim: int = 64,
                    strategy_holder: Optional[dict] = None):
    def server_fn(context: Context) -> ServerAppComponents:
        init_model = build_model(input_dim=input_dim, feature_dim=feature_dim, n_classes=num_classes)
        init_parameters = ndarrays_to_parameters(get_parameters(init_model))

        def fit_config(server_round: int) -> Dict[str, Scalar]:
            return {"local_epochs": LOCAL_EPOCHS, "lr": LR, "server_round": server_round}

        strategy = RecordingFedAvg(
            fraction_fit=FRACTION_FIT,
            fraction_evaluate=FRACTION_EVALUATE,
            min_fit_clients=max(1, int(N_CLIENTS * FRACTION_FIT)),
            min_evaluate_clients=max(1, int(N_CLIENTS * FRACTION_EVALUATE)),
            min_available_clients=N_CLIENTS,
            initial_parameters=init_parameters,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=weighted_average,
        )

        if strategy_holder is not None:
            strategy_holder["strategy"] = strategy

        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn


# =========================================================
# FINAL CENTRALIZED-STYLE TEST EVAL (optional, after training)
# =========================================================
def final_test_evaluation(clients, input_dim: int, num_classes: int,
                            final_parameters: NDArrays, feature_dim: int = 64) -> None:
    """Evaluate the final global model on each client's held-out test set
    and print a pooled classification report."""
    all_preds, all_targets = [], []

    for cid in range(N_CLIENTS):
        (_, _), (_, _), (X_test, y_test) = get_client_data(clients, cid)
        test_loader = to_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

        model = build_model(input_dim=input_dim, feature_dim=feature_dim, n_classes=num_classes).to(DEVICE)
        set_parameters(model, final_parameters)
        model.eval()

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                preds = torch.argmax(model(xb), dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(yb.numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    print("\n=== FINAL GLOBAL MODEL — POOLED TEST METRICS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")


# =========================================================
# ENTRY POINT
# =========================================================
def run_federated_learning(clients, input_dim: int, num_classes: int,
                             num_rounds: int = NUM_ROUNDS,
                             n_clients: int = N_CLIENTS,
                             feature_dim: int = 64):
    """
    clients    : object returned by make_federated_splits(...)
    input_dim  : number of input features fed to build_model()
    num_classes: number of output classes (-> build_model's n_classes)
    feature_dim: size of the shared feature representation (theta output)
    """
    global N_CLIENTS
    N_CLIENTS = n_clients

    strategy_holder: Dict[str, RecordingFedAvg] = {}

    client_app = ClientApp(client_fn=make_client_fn(clients, input_dim, num_classes, feature_dim))
    server_app = ServerApp(
        server_fn=make_server_fn(input_dim, num_classes, num_rounds, feature_dim, strategy_holder)
    )

    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    if DEVICE.type == "cuda":
        backend_config["client_resources"]["num_gpus"] = 1.0 / max(n_clients, 1)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=n_clients,
        backend_config=backend_config,
    )

    # run_simulation() itself returns None in this Flower API style —
    # the real per-round metrics live on the strategy we stashed above.
    strategy = strategy_holder.get("strategy")
    return strategy.history if strategy is not None else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--clients", type=int, default=N_CLIENTS)
    parser.add_argument("--input_dim", type=int, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    args = parser.parse_args()

    # Example of how this ties into main.ipynb:
    #
    #   from make_federated_splits import make_federated_splits
    #   from fed import run_federated_learning
    #
    #   clients = make_federated_splits(
    #       X=X, y=y, n_clients=4,
    #       mode="non_iid", alpha=0.5,
    #       test_size=0.15, val_size=0.15, seed=42,
    #       oversample_method="smote",
    #   )
    #
    #   history = run_federated_learning(
    #       clients,
    #       input_dim=X.shape[1],
    #       num_classes=len(set(y)),
    #       num_rounds=20,
    #       n_clients=4,
    #   )
    #
    # This __main__ block is just a CLI smoke test placeholder; in
    # practice you'll typically call run_federated_learning(...) directly
    # from main.ipynb with `clients` already in memory.
    print(
        "This script defines run_federated_learning(clients, input_dim, num_classes, ...). "
        "Import it and call that function from main.ipynb after building `clients` with "
        "make_federated_splits()."
    )
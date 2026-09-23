"""
fl_runner.py

Simulasi Federated Learning (Flower) untuk klasifikasi Drebin.
Varian PERSONALIZED (FedPer): hanya theta (FeatureExtractor) yang diagregasi
di server. phi (ClassifierHead) tetap lokal di tiap client dan tidak pernah
dikirim ke server.

Bandwidth dihitung dari ukuran payload theta saja (nbytes array numpy).
"""

import os
import shutil
import tempfile
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, Context, NDArrays
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MB = 1024 ** 2
METRIC_KEYS = ("accuracy", "precision", "recall", "f1")


# -----------------------------------------------------------------------------
# Utilitas parameter (hanya THETA yang dipertukarkan)
# -----------------------------------------------------------------------------
def get_theta_params(model: nn.Module) -> NDArrays:
    return [v.detach().cpu().numpy() for v in model.theta.state_dict().values()]


def set_theta_params(model: nn.Module, params: NDArrays) -> None:
    keys = model.theta.state_dict().keys()
    state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)})
    model.theta.load_state_dict(state, strict=True)


def params_nbytes(params: NDArrays) -> int:
    return int(sum(np.asarray(p).nbytes for p in params))


def module_nbytes(module: nn.Module) -> int:
    return int(sum(v.numel() * v.element_size()
                   for v in module.state_dict().values()))


def to_tensors(X, y):
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    return (torch.as_tensor(np.asarray(X), dtype=torch.float32),
            torch.as_tensor(np.asarray(y), dtype=torch.long))


# -----------------------------------------------------------------------------
# Penyimpanan phi lokal per client (persisten antar ronde)
# -----------------------------------------------------------------------------
def head_path(head_dir: str, cid: int) -> str:
    return os.path.join(head_dir, f"phi_client_{cid}.pt")


def save_head(model: nn.Module, head_dir: str, cid: int) -> None:
    torch.save({k: v.cpu() for k, v in model.phi.state_dict().items()},
               head_path(head_dir, cid))


def load_head_if_exists(model: nn.Module, head_dir: str, cid: int) -> bool:
    p = head_path(head_dir, cid)
    if os.path.exists(p):
        model.phi.load_state_dict(torch.load(p, map_location="cpu"))
        return True
    return False


# -----------------------------------------------------------------------------
# Train / evaluasi
# -----------------------------------------------------------------------------
def train_local(model, loader, epochs, lr, proximal_mu=0.0, global_theta=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            if proximal_mu > 0 and global_theta is not None:  # FedProx (theta saja)
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.theta.parameters(), global_theta))
                loss = loss + (proximal_mu / 2) * prox
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate_model(model, X, y):
    model.eval()
    X, y = to_tensors(X, y)
    logits = model(X.to(DEVICE))
    loss = nn.CrossEntropyLoss()(logits, y.to(DEVICE)).item()
    pred = logits.argmax(1).cpu().numpy()
    yt = y.numpy()
    return loss, {
        "accuracy": float((pred == yt).mean()),
        "precision": float(precision_score(yt, pred, zero_division=0)),
        "recall": float(recall_score(yt, pred, zero_division=0)),
        "f1": float(f1_score(yt, pred, zero_division=0)),
    }


def evaluate_personalized(theta_params, X, y, n_features, n_classes,
                          head_dir, num_clients):
    """
    theta global + phi milik tiap client -> evaluasi di (X, y) -> rata-rata.
    Return: (mean_loss, mean_metrics, per_client_metrics)
    """
    losses, per_client = [], []
    for cid in range(num_clients):
        model = build_model(n_features, n_classes).to(DEVICE)
        set_theta_params(model, theta_params)
        if not load_head_if_exists(model, head_dir, cid):
            continue  # client ini belum pernah training
        model.to(DEVICE)
        loss, m = evaluate_model(model, X, y)
        losses.append(loss)
        per_client.append(m)
    if not per_client:
        return None, None, []
    mean_m = {k: float(np.mean([m[k] for m in per_client])) for k in METRIC_KEYS}
    return float(np.mean(losses)), mean_m, per_client


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------
class FlowerClient(NumPyClient):
    def __init__(self, cid, X, y, n_features, n_classes, local_epochs,
                 batch_size, lr, head_dir, proximal_mu=0.0):
        self.cid = cid
        self.X, self.y = to_tensors(X, y)
        self.model = build_model(n_features, n_classes).to(DEVICE)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.head_dir = head_dir
        self.proximal_mu = proximal_mu

    def _restore_local_head(self):
        if load_head_if_exists(self.model, self.head_dir, self.cid):
            self.model.to(DEVICE)

    def fit(self, parameters, config):
        bytes_down = params_nbytes(parameters)          # theta dari server

        set_theta_params(self.model, parameters)        # theta <- global
        self._restore_local_head()                      # phi   <- lokal (ronde lalu)
        global_theta = [torch.tensor(p).to(DEVICE) for p in parameters]

        loader = DataLoader(TensorDataset(self.X, self.y),
                            batch_size=self.batch_size, shuffle=True)
        train_local(self.model, loader, self.local_epochs, self.lr,
                    self.proximal_mu, global_theta)

        save_head(self.model, self.head_dir, self.cid)  # phi tetap lokal

        new_params = get_theta_params(self.model)       # hanya theta dikirim
        bytes_up = params_nbytes(new_params)
        return new_params, len(self.X), {
            "bytes_down": float(bytes_down),
            "bytes_up": float(bytes_up),
        }

    def evaluate(self, parameters, config):
        set_theta_params(self.model, parameters)
        self._restore_local_head()
        loss, m = evaluate_model(self.model, self.X, self.y)
        return loss, len(self.X), m


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_federated(
    fed_data,
    name: str,
    n_features: int,
    n_classes: int = 2,
    X_val=None, y_val=None,
    X_test=None, y_test=None,
    proximal_mu: float = 0.0,
    num_rounds: int = 10,
    local_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Dict:
    """
    Jalankan satu skenario FL (hanya theta yang diagregasi).

    History per ronde: round, loss, accuracy, precision, recall, f1
      -> metrik rata-rata dari (theta global + phi tiap client) pada X_val.
    Bandwidth: bytes_up, bytes_down, bytes_total, cum_mb, comm (ringkasan).
    history["test"]            : rata-rata metrik test (ronde terakhir)
    history["test_per_client"] : metrik test per client
    """
    num_clients = len(fed_data)
    head_dir = tempfile.mkdtemp(prefix="fl_heads_")

    history: Dict = {k: [] for k in
                     ["round", "loss", "accuracy", "precision", "recall", "f1",
                      "bytes_up", "bytes_down", "bytes_total", "cum_mb"]}
    last = {"params": None}

    def client_fn(context: Context) -> Client:
        cid = int(context.node_config["partition-id"])
        c = fed_data[cid]
        return FlowerClient(cid, c["X_train"], c["y_train"], n_features,
                            n_classes, local_epochs, batch_size, lr,
                            head_dir, proximal_mu).to_client()

    def fit_metrics_agg(results):
        up = sum(m.get("bytes_up", 0.0) for _, m in results)
        down = sum(m.get("bytes_down", 0.0) for _, m in results)
        prev = history["cum_mb"][-1] if history["cum_mb"] else 0.0
        history["bytes_up"].append(up)
        history["bytes_down"].append(down)
        history["bytes_total"].append(up + down)
        history["cum_mb"].append(prev + (up + down) / MB)
        return {"bytes_up": up, "bytes_down": down}

    def evaluate_fn(server_round, parameters, config):
        if server_round == 0:
            return None
        theta = [np.asarray(a) for a in
                 __import__("flwr").common.parameters_to_ndarrays(parameters)] \
            if not isinstance(parameters, list) else parameters
        loss, m, _ = evaluate_personalized(
            theta, X_val, y_val, n_features, n_classes, head_dir, num_clients)
        if m is None:
            return None
        last["params"] = theta
        history["round"].append(server_round)
        history["loss"].append(loss)
        for k in METRIC_KEYS:
            history[k].append(m[k])
        comm = ""
        if len(history["bytes_total"]) >= server_round:
            comm = (f" | comm={history['bytes_total'][server_round - 1] / MB:.3f}MB"
                    f" (cum {history['cum_mb'][server_round - 1]:.3f}MB)")
        print(f"[{name}] round {server_round:02d} | loss={loss:.4f} "
              f"acc={m['accuracy']:.4f} f1={m['f1']:.4f}{comm}")
        return loss, m

    # parameter awal = theta saja
    init_model = build_model(n_features, n_classes)
    init_params = ndarrays_to_parameters(get_theta_params(init_model))

    def server_fn(context: Context) -> ServerAppComponents:
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=init_params,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=fit_metrics_agg,
        )
        return ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

    try:
        run_simulation(
            server_app=ServerApp(server_fn=server_fn),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=num_clients,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
        )

        # evaluasi akhir di test set: theta global ronde terakhir + phi tiap client
        if X_test is not None and last["params"] is not None:
            _, mean_m, per_client = evaluate_personalized(
                last["params"], X_test, y_test, n_features, n_classes,
                head_dir, num_clients)
            history["test"] = mean_m
            history["test_per_client"] = per_client
    finally:
        shutil.rmtree(head_dir, ignore_errors=True)

    # ringkasan bandwidth
    total_up = float(sum(history["bytes_up"]))
    total_down = float(sum(history["bytes_down"]))
    theta_bytes = params_nbytes(get_theta_params(init_model))
    full_bytes = module_nbytes(init_model)
    n_rounds_done = max(len(history["bytes_total"]), 1)
    history["comm"] = {
        "model_size_mb": full_bytes / MB,          # theta + phi (referensi FedAvg penuh)
        "theta_size_mb": theta_bytes / MB,         # yang benar-benar dikirim
        "saving_vs_full_pct": 100.0 * (1 - theta_bytes / full_bytes),
        "total_up_mb": total_up / MB,
        "total_down_mb": total_down / MB,
        "total_mb": (total_up + total_down) / MB,
        "avg_per_round_mb": (total_up + total_down) / MB / n_rounds_done,
        "avg_per_client_mb": (total_up + total_down) / MB / max(num_clients, 1),
    }
    c = history["comm"]
    print(f"[{name}] BANDWIDTH | theta={c['theta_size_mb']:.3f}MB "
          f"(full={c['model_size_mb']:.3f}MB, hemat {c['saving_vs_full_pct']:.2f}%) "
          f"up={c['total_up_mb']:.2f}MB down={c['total_down_mb']:.2f}MB "
          f"total={c['total_mb']:.2f}MB (~{c['avg_per_round_mb']:.2f}MB/ronde)")

    return history
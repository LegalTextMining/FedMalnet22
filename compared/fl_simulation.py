"""
fl_runner.py

Simulasi Federated Learning (Flower) untuk klasifikasi Drebin.
Semua dependensi (n_features, n_classes, dst.) dikirim lewat argumen,
tidak bergantung pada variabel global notebook.

Tambahan: perhitungan bandwidth (biaya komunikasi) per ronde.
  - downlink : server -> client (bobot global yang dikirim ke tiap client)
  - uplink   : client -> server (bobot lokal hasil training)
Dihitung dari ukuran payload array numpy (nbytes) sehingga overhead
serialisasi/protobuf gRPC tidak ikut terhitung (biasanya sangat kecil).
"""

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


# -----------------------------------------------------------------------------
# Utilitas parameter & data
# -----------------------------------------------------------------------------
def get_params(model: nn.Module) -> NDArrays:
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_params(model: nn.Module, params: NDArrays) -> None:
    keys = model.state_dict().keys()
    state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)})
    model.load_state_dict(state, strict=True)


def params_nbytes(params: NDArrays) -> int:
    """Total ukuran payload (byte) dari list array numpy."""
    return int(sum(np.asarray(p).nbytes for p in params))


def to_tensors(X, y):
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    return (torch.as_tensor(np.asarray(X), dtype=torch.float32),
            torch.as_tensor(np.asarray(y), dtype=torch.long))


# -----------------------------------------------------------------------------
# Train / evaluasi
# -----------------------------------------------------------------------------
def train_local(model, loader, epochs, lr, proximal_mu=0.0, global_params=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            if proximal_mu > 0 and global_params is not None:  # FedProx
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), global_params))
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


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------
class FlowerClient(NumPyClient):
    def __init__(self, X, y, n_features, n_classes, local_epochs,
                 batch_size, lr, proximal_mu=0.0):
        self.X, self.y = to_tensors(X, y)
        self.model = build_model(n_features, n_classes).to(DEVICE)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.proximal_mu = proximal_mu

    def fit(self, parameters, config):
        # bandwidth: bobot global yang DITERIMA client (downlink)
        bytes_down = params_nbytes(parameters)

        set_params(self.model, parameters)
        global_params = [torch.tensor(p).to(DEVICE) for p in parameters]
        loader = DataLoader(TensorDataset(self.X, self.y),
                            batch_size=self.batch_size, shuffle=True)
        train_local(self.model, loader, self.local_epochs, self.lr,
                    self.proximal_mu, global_params)

        new_params = get_params(self.model)
        # bandwidth: bobot lokal yang DIKIRIM client (uplink)
        bytes_up = params_nbytes(new_params)

        # metrik dikirim ke server lewat dict (harus skalar)
        return new_params, len(self.X), {
            "bytes_down": float(bytes_down),
            "bytes_up": float(bytes_up),
        }

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
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
) -> Dict[str, List[float]]:
    """
    Jalankan satu skenario FL. Mengembalikan dict history per round
    (round, loss, accuracy, precision, recall, f1) + key "test" berisi
    metrik test set dari model global ronde terakhir (jika X_test diberikan).

    Tambahan bandwidth:
      - history["bytes_up"], ["bytes_down"], ["bytes_total"] : per ronde (byte,
        dijumlahkan seluruh client)
      - history["cum_mb"] : kumulatif total (uplink + downlink) dalam MB
      - history["comm"]   : ringkasan total (MB) di akhir simulasi
    """
    num_clients = len(fed_data)
    history: Dict = {k: [] for k in
                     ["round", "loss", "accuracy", "precision", "recall", "f1",
                      "bytes_up", "bytes_down", "bytes_total", "cum_mb"]}
    last = {"params": None}

    def client_fn(context: Context) -> Client:
        cid = int(context.node_config["partition-id"])
        c = fed_data[cid]
        return FlowerClient(c["X_train"], c["y_train"], n_features, n_classes,
                            local_epochs, batch_size, lr, proximal_mu).to_client()

    # Dipanggil server SETIAP ronde setelah fit client selesai
    # (sebelum evaluate_fn ronde yang sama).
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
        if server_round == 0:          # evaluasi bobot awal, lewati
            return None
        model = build_model(n_features, n_classes).to(DEVICE)
        set_params(model, parameters)
        loss, m = evaluate_model(model, X_val, y_val)
        last["params"] = parameters
        history["round"].append(server_round)
        history["loss"].append(loss)
        for k in ("accuracy", "precision", "recall", "f1"):
            history[k].append(m[k])
        comm = ""
        if len(history["bytes_total"]) >= server_round:
            comm = (f" | comm={history['bytes_total'][server_round - 1] / MB:.3f}MB"
                    f" (cum {history['cum_mb'][server_round - 1]:.3f}MB)")
        print(f"[{name}] round {server_round:02d} | loss={loss:.4f} "
              f"acc={m['accuracy']:.4f} f1={m['f1']:.4f}{comm}")
        return loss, m

    init_params = ndarrays_to_parameters(
        get_params(build_model(n_features, n_classes)))

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

    run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=ClientApp(client_fn=client_fn),
        num_supernodes=num_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )

    # evaluasi akhir di test set memakai model global ronde terakhir
    if X_test is not None and last["params"] is not None:
        model = build_model(n_features, n_classes).to(DEVICE)
        set_params(model, last["params"])
        _, history["test"] = evaluate_model(model, X_test, y_test)

    # ringkasan bandwidth
    total_up = float(sum(history["bytes_up"]))
    total_down = float(sum(history["bytes_down"]))
    model_bytes = params_nbytes(get_params(build_model(n_features, n_classes)))
    n_rounds_done = max(len(history["bytes_total"]), 1)
    history["comm"] = {
        "model_size_mb": model_bytes / MB,
        "total_up_mb": total_up / MB,
        "total_down_mb": total_down / MB,
        "total_mb": (total_up + total_down) / MB,
        "avg_per_round_mb": (total_up + total_down) / MB / n_rounds_done,
        "avg_per_client_mb": (total_up + total_down) / MB / max(num_clients, 1),
    }
    c = history["comm"]
    print(f"[{name}] BANDWIDTH | model={c['model_size_mb']:.3f}MB "
          f"up={c['total_up_mb']:.2f}MB down={c['total_down_mb']:.2f}MB "
          f"total={c['total_mb']:.2f}MB "
          f"(~{c['avg_per_round_mb']:.2f}MB/ronde)")

    return history
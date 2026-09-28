"""
fl_runner.py
"""
import os
import shutil
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Context, NDArrays
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from model import build_model

DEVICE = torch.device("cpu")

MB = 1024 ** 2
METRIC_KEYS = ("accuracy", "precision", "recall", "f1")


# -----------------------------------------------------------------------------
# Wrapper: Adapter (lokal) -> theta (dibagi) -> phi (lokal)
# -----------------------------------------------------------------------------
class ClientModel(nn.Module):
    """
    Membungkus base model (dari build_model) dengan adapter input lokal
    agar client dengan jumlah fitur berbeda tetap bisa berbagi `theta`
    yang beroperasi pada dimensi bersama (common_dim).
    """

    def __init__(self, n_features_raw: int, common_dim: int, n_classes: int):
        super().__init__()
        self.adapter = nn.Linear(n_features_raw, common_dim)
        self.base = build_model(common_dim, n_classes)  # punya .theta dan .phi
        if not hasattr(self.base, "theta") or not hasattr(self.base, "phi"):
            raise AttributeError(
                "build_model(...) harus mengembalikan model dengan atribut "
                "`.theta` (FeatureExtractor) dan `.phi` (ClassifierHead)."
            )

    @property
    def theta(self) -> nn.Module:
        return self.base.theta

    @property
    def phi(self) -> nn.Module:
        return self.base.phi

    def forward(self, x):
        z = torch.relu(self.adapter(x))
        feat = self.theta(z)
        return self.phi(feat)


# -----------------------------------------------------------------------------
# Utilitas parameter (hanya THETA yang dipertukarkan)
# -----------------------------------------------------------------------------
def get_theta_params(model: ClientModel) -> NDArrays:
    return [v.detach().cpu().numpy() for v in model.theta.state_dict().values()]


def set_theta_params(model: ClientModel, params: NDArrays) -> None:
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
# Penyimpanan bagian LOKAL (adapter + phi) per client (persisten antar ronde)
# -----------------------------------------------------------------------------
def local_path(local_dir: str, cid: int) -> str:
    return os.path.join(local_dir, f"local_client_{cid}.pt")


def save_local(model: ClientModel, local_dir: str, cid: int) -> None:
    torch.save({
        "adapter": {k: v.cpu() for k, v in model.adapter.state_dict().items()},
        "phi": {k: v.cpu() for k, v in model.phi.state_dict().items()},
    }, local_path(local_dir, cid))


def load_local_if_exists(model: ClientModel, local_dir: str, cid: int) -> bool:
    p = local_path(local_dir, cid)
    if os.path.exists(p):
        blob = torch.load(p, map_location="cpu")
        model.adapter.load_state_dict(blob["adapter"])
        model.phi.load_state_dict(blob["phi"])
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


def evaluate_personalized_per_client(
    theta_params: NDArrays,
    fed_data: Dict[str, dict],
    common_dim: int,
    n_classes: int,
    local_dir: str,
    split: str,  # "val" atau "test"
):
    """
    theta global + (adapter, phi) milik tiap client -> evaluasi di data
    milik client itu sendiri (karena tiap client = dataset berbeda,
    dengan jumlah fitur berbeda).

    Return: (mean_loss, mean_metrics, {nama_client: metrics})
    """
    losses, per_client = [], {}
    for cid, (name, c) in enumerate(fed_data.items()):
        Xk, yk = c.get(f"X_{split}"), c.get(f"y_{split}")
        if Xk is None or yk is None:
            continue
        n_features_raw = c["n_features"]
        model = ClientModel(n_features_raw, common_dim, n_classes).to(DEVICE)
        set_theta_params(model, theta_params)
        if not load_local_if_exists(model, local_dir, cid):
            continue  # client ini belum pernah training (adapter/phi random)
        loss, m = evaluate_model(model, Xk, yk)
        losses.append(loss)
        per_client[name] = m
    if not per_client:
        return None, None, {}
    mean_m = {k: float(np.mean([m[k] for m in per_client.values()]))
              for k in METRIC_KEYS}
    return float(np.mean(losses)), mean_m, per_client


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------
class FlowerClient(NumPyClient):
    def __init__(self, cid, name, X, y, n_features_raw, common_dim, n_classes,
                 local_epochs, batch_size, lr, local_dir, proximal_mu=0.0):
        self.cid = cid
        self.name = name
        self.X, self.y = to_tensors(X, y)
        self.model = ClientModel(n_features_raw, common_dim, n_classes).to(DEVICE)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.local_dir = local_dir
        self.proximal_mu = proximal_mu

    def _restore_local(self):
        if load_local_if_exists(self.model, self.local_dir, self.cid):
            self.model.to(DEVICE)

    def fit(self, parameters, config):
        bytes_down = params_nbytes(parameters)          # theta dari server

        set_theta_params(self.model, parameters)        # theta <- global
        self._restore_local()                           # adapter+phi <- lokal (ronde lalu)
        global_theta = [torch.tensor(p).to(DEVICE) for p in parameters]

        loader = DataLoader(TensorDataset(self.X, self.y),
                            batch_size=self.batch_size, shuffle=True)
        train_local(self.model, loader, self.local_epochs, self.lr,
                    self.proximal_mu, global_theta)

        save_local(self.model, self.local_dir, self.cid)  # adapter+phi tetap lokal

        new_params = get_theta_params(self.model)          # hanya theta dikirim
        bytes_up = params_nbytes(new_params)
        return new_params, len(self.X), {
            "bytes_down": float(bytes_down),
            "bytes_up": float(bytes_up),
        }

    def evaluate(self, parameters, config):
        set_theta_params(self.model, parameters)
        self._restore_local()
        loss, m = evaluate_model(self.model, self.X, self.y)
        return loss, len(self.X), m

import csv
# from aggregation import build_fedavg_strategy
from aggregationnew import build_fedcosine_pairwise_strategy

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_federated(
    fed_data: Dict[str, dict],
    name: str,
    common_dim: int,
    n_classes: int = 2,
    proximal_mu: float = 0.0,
    num_rounds: int = 10,
    local_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Dict:
    for cname, c in fed_data.items():
        if "n_features" not in c:
            raise KeyError(f"fed_data['{cname}'] harus punya key 'n_features'.")

    num_clients = len(fed_data)
    client_names = list(fed_data.keys())
    local_dir = tempfile.mkdtemp(prefix="fl_local_")

    history: Dict = {k: [] for k in
                     ["round", "loss", "accuracy", "precision", "recall", "f1",
                      "bytes_up", "bytes_down", "bytes_total", "cum_mb",
                      "val_per_client"]}
    last = {"params": None}

    def client_fn(context: Context) -> Client:
        cid = int(context.node_config["partition-id"])
        cname = client_names[cid]
        c = fed_data[cname]
        return FlowerClient(
            cid, cname, c["X_train"], c["y_train"], c["n_features"],
            common_dim, n_classes, local_epochs, batch_size, lr,
            local_dir, proximal_mu,
        ).to_client()

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
        theta = parameters if isinstance(parameters, list) \
            else [np.asarray(a) for a in parameters_to_ndarrays(parameters)]
        loss, m, per_client = evaluate_personalized_per_client(
            theta, fed_data, common_dim, n_classes, local_dir, split="val")
        if m is None:
            return None
        last["params"] = theta
        history["round"].append(server_round)
        history["loss"].append(loss)
        history["val_per_client"].append(per_client)
        for k in METRIC_KEYS:
            history[k].append(m[k])
        comm = ""
        if len(history["bytes_total"]) >= server_round:
            comm = (f" | comm={history['bytes_total'][server_round - 1] / MB:.3f}MB"
                    f" (cum {history['cum_mb'][server_round - 1]:.3f}MB)")
        print(f"[{name}] round {server_round:02d} | loss={loss:.4f} "
              f"acc={m['accuracy']:.4f} f1={m['f1']:.4f}{comm}")
        return loss, m

    # parameter awal = theta saja (dimensi common_dim, sama untuk semua client)
    init_model = ClientModel(common_dim, common_dim, n_classes)  # n_features dummy
    init_params = ndarrays_to_parameters(get_theta_params(init_model))
    def weighted_average(metrics):
    # fit_metrics_aggregation_fn kamu yang sudah ada
        total = sum(n for n, _ in metrics)
        acc = sum(n * m.get("accuracy", 0.0) for n, m in metrics) / total
        return {"accuracy": acc}

    def server_fn(context: Context) -> ServerAppComponents:
        strategy = build_fedcosine_pairwise_strategy(
        num_clients=num_clients,
        init_params=init_params,
        evaluate_fn=evaluate_fn,
        fit_metrics_agg=weighted_average,
        temperature=0.05,
        min_weight=0.0,
        similarity_agg="mean",          # atau "median"
        data_weight_influence=0.3,      # similarity tetap dominan
        combine_mode="geometric",       # atau "linear"
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

        # evaluasi akhir di test set: theta global ronde terakhir + adapter/phi tiap client
        if last["params"] is not None:
            _, mean_m, per_client = evaluate_personalized_per_client(
                last["params"], fed_data, common_dim, n_classes, local_dir,
                split="test")
            history["test"] = mean_m
            history["test_per_client"] = per_client
    finally:
        shutil.rmtree(local_dir, ignore_errors=True)

    # ringkasan bandwidth (hanya theta yang lewat jaringan)
    total_up = float(sum(history["bytes_up"]))
    total_down = float(sum(history["bytes_down"]))
    theta_bytes = params_nbytes(get_theta_params(init_model))
    n_rounds_done = max(len(history["bytes_total"]), 1)
    history["comm"] = {
        "theta_size_mb": theta_bytes / MB,
        "total_up_mb": total_up / MB,
        "total_down_mb": total_down / MB,
        "total_mb": (total_up + total_down) / MB,
        "avg_per_round_mb": (total_up + total_down) / MB / n_rounds_done,
        "avg_per_client_mb": (total_up + total_down) / MB / max(num_clients, 1),
    }
    c = history["comm"]
    print(f"[{name}] BANDWIDTH | theta={c['theta_size_mb']:.3f}MB "
          f"up={c['total_up_mb']:.2f}MB down={c['total_down_mb']:.2f}MB "
          f"total={c['total_mb']:.2f}MB (~{c['avg_per_round_mb']:.2f}MB/ronde)")

    return history

def export_history_csv(history: Dict, out_path: str) -> str:
    """
    Ekspor metrik per-ronde (loss, accuracy, precision, recall, f1)
    beserta bandwidth (bytes_up, bytes_down, bytes_total, cum_mb) ke CSV.
    """
    rounds = history.get("round", [])
    n = len(rounds)
    fieldnames = [
        "round", "loss", "accuracy", "precision", "recall", "f1",
        "bytes_up_MB", "bytes_down_MB", "bytes_total_MB", "cum_MB",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "round": history["round"][i],
                "loss": history["loss"][i],
                "accuracy": history["accuracy"][i],
                "precision": history["precision"][i],
                "recall": history["recall"][i],
                "f1": history["f1"][i],
                "bytes_up_MB": (history["bytes_up"][i] / MB
                                if i < len(history["bytes_up"]) else ""),
                "bytes_down_MB": (history["bytes_down"][i] / MB
                                  if i < len(history["bytes_down"]) else ""),
                "bytes_total_MB": (history["bytes_total"][i] / MB
                                   if i < len(history["bytes_total"]) else ""),
                "cum_MB": (history["cum_mb"][i]
                           if i < len(history["cum_mb"]) else ""),
            })
    print(f"[export] metrik per-ronde tersimpan di: {out_path}")
    return out_path


def export_per_client_csv(history: Dict, out_path: str, split: str = "val_per_client") -> str:
    """
    Ekspor metrik per-client per-ronde (accuracy, precision, recall, f1)
    ke CSV, berguna untuk melihat performa personalisasi tiap client.
    """
    rows = []
    for r_idx, per_client in zip(history["round"], history.get(split, [])):
        for cname, m in per_client.items():
            rows.append({
                "round": r_idx, "client": cname,
                "accuracy": m["accuracy"], "precision": m["precision"],
                "recall": m["recall"], "f1": m["f1"],
            })
    if not rows:
        return ""
    fieldnames = ["round", "client", "accuracy", "precision", "recall", "f1"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[export] metrik per-client tersimpan di: {out_path}")
    return out_path
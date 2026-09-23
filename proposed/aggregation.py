from flwr.server.strategy import FedAvg

"""
Strategy Flower kustom: agregasi berbasis cosine distance ke theta global,
bukan berbasis num_examples (FedAvg standar).

Client dengan theta yang cosine-similarity-nya lebih tinggi ke theta global
ronde sebelumnya akan mendapat bobot agregasi lebih besar.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from flwr.common import (
    FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg



def build_fedavg_strategy(
    num_clients,
    init_params,
    evaluate_fn,
    fit_metrics_agg,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.0,
):
    """Buat strategy FedAvg dengan konfigurasi standar untuk simulasi ini."""
    return FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )

def build_fedcosine_strategy(num_clients, init_params, evaluate_fn,
                              fit_metrics_agg, temperature=0.1, min_weight=0.0):
    return FedCosine(
        temperature=temperature,
        min_weight=min_weight,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )

# def build_distanceAG_strategy(
        
# ):
def _flatten(ndarrays: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(p).flatten() for p in ndarrays])


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FedCosine(FedAvg):
    """
    FedAvg, tapi bobot agregasi per client = softmax(cosine_similarity ke theta
    global ronde sebelumnya), bukan proporsional num_examples.

    temperature: mengatur seberapa 'tajam' perbedaan bobot antar client.
        - temperature kecil (mis. 0.05) -> perbedaan bobot makin ekstrem
          (client paling mirip global mendominasi).
        - temperature besar (mis. 1.0)  -> bobot makin merata (mendekati rata2 biasa).
    min_weight: batas bawah bobot supaya client yang menyimpang tidak dapat
        bobot 0 total (opsional, untuk stabilitas / fairness).
    """

    def __init__(self, *args, temperature: float = 0.1,
                 min_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.min_weight = min_weight
        self._prev_global_flat: Optional[np.ndarray] = None  # theta global ronde sblmnya

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        if not results:
            return None, {}

        # ambil theta tiap client (ndarrays) + metrics
        client_thetas: List[np.ndarray] = []
        client_ndarrays: List[List[np.ndarray]] = []
        for _, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            client_ndarrays.append(nds)
            client_thetas.append(_flatten(nds))

        # referensi: theta global ronde sebelumnya. Kalau belum ada (ronde 1),
        # pakai rata-rata theta client ronde ini sebagai referensi sementara.
        if self._prev_global_flat is None:
            ref = np.mean(np.stack(client_thetas, axis=0), axis=0)
        else:
            ref = self._prev_global_flat

        # hitung cosine similarity tiap client ke referensi
        sims = np.array([cosine_similarity(t, ref) for t in client_thetas])

        # ubah similarity jadi bobot lewat softmax (biar semua positif & jumlah=1)
        # similarity range [-1, 1] -> dibagi temperature lalu softmax
        scaled = sims / max(self.temperature, 1e-6)
        scaled = scaled - scaled.max()  # stabilitas numerik
        weights = np.exp(scaled)
        weights = weights / weights.sum()

        # opsional: pasang batas bawah bobot, lalu renormalisasi
        if self.min_weight > 0:
            weights = np.maximum(weights, self.min_weight)
            weights = weights / weights.sum()

        # weighted average per layer (bukan pakai num_examples seperti FedAvg asli)
        n_layers = len(client_ndarrays[0])
        agg_ndarrays = []
        for layer_idx in range(n_layers):
            layer_stack = np.stack(
                [client_ndarrays[i][layer_idx] for i in range(len(client_ndarrays))],
                axis=0,
            )
            w = weights.reshape([-1] + [1] * (layer_stack.ndim - 1))
            agg_layer = np.sum(layer_stack * w, axis=0)
            agg_ndarrays.append(agg_layer.astype(client_ndarrays[0][layer_idx].dtype))

        # simpan sebagai referensi untuk ronde berikutnya
        self._prev_global_flat = _flatten(agg_ndarrays)

        aggregated_params = ndarrays_to_parameters(agg_ndarrays)

        # metrics tambahan biar bisa dipantau: bobot & similarity tiap client
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        # tempel info cosine ke metrics (opsional, buat tracking/report)
        for i, (client_proxy, fit_res) in enumerate(results):
            cname = fit_res.metrics.get("client_name", f"client_{i}")
            metrics_aggregated[f"cos_sim_{cname}"] = float(sims[i])
            metrics_aggregated[f"agg_weight_{cname}"] = float(weights[i])

        return aggregated_params, metrics_aggregated
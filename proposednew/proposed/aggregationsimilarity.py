"""
FedCosinePairwise Strategy
===========================
Custom FedAvg: bobot per client = kombinasi
    - similarity antar representasi (parameter) client (FAKTOR UTAMA)
    - jumlah data / num_examples (FAKTOR SEKUNDER)

data_weight_influence (0-1) mengatur porsi pengaruh jumlah data;
default 0.3 -> similarity tetap dominan (70% similarity, 30% data).

combine_mode:
    "linear"          -> weights = (1-a)*sim_w + a*data_w
    "geometric"       -> weights = sim_w^(1-a) * data_w^a  (default,
                         similarity rendah tidak bisa "diselamatkan"
                         oleh data besar karena perkalian, bukan
                         penjumlahan)
    "similarity_only" -> weights = sim_w saja. Jumlah data (num_examples)
                         diabaikan TOTAL, data_weight_influence tidak
                         dipakai sama sekali di mode ini.
"""

from typing import Dict, List, Optional
import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# ============================================================
# Utility
# ============================================================
def _flatten_all(ndarrays_list: List[List[np.ndarray]]) -> np.ndarray:
    """Flatten parameter tiap client -> matrix (n_client, n_param)."""
    return np.stack(
        [
            np.concatenate([np.asarray(p).flatten() for p in nds])
            for nds in ndarrays_list
        ]
    )


def _pairwise_cosine_matrix(thetas: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Cosine similarity matrix (n x n), vectorized (ganti double-loop)."""
    norms = np.linalg.norm(thetas, axis=1, keepdims=True)
    norms = np.where(norms < eps, eps, norms)
    normalized = thetas / norms
    return normalized @ normalized.T


# ============================================================
# Builder
# ============================================================
def build_fedcosine_pairwise_strategy(
    num_clients: int,
    init_params: Parameters,
    evaluate_fn,
    fit_metrics_agg,
    temperature: float = 0.1,
    min_weight: float = 0.0,
    similarity_agg: str = "mean",
    data_weight_influence: float = 0.3,
    combine_mode: str = "geometric",
) -> "FedCosinePairwise":
    """
    temperature: ketajaman softmax similarity.
    min_weight: bobot minimum (floor) per client, diterapkan setelah kombinasi.
    similarity_agg: "mean" atau "median" similarity client ke client lain
        ("median" lebih tahan client outlier/mencurigakan).
    data_weight_influence: 0 -> bobot murni similarity, 1 -> murni proporsi
        data. Diabaikan (tidak dipakai) kalau combine_mode="similarity_only".
    combine_mode: "linear", "geometric", atau "similarity_only"
        (lihat docstring modul). Pakai "similarity_only" kalau jumlah
        data client memang ingin diabaikan total.
    """
    return FedCosinePairwise(
        temperature=temperature,
        min_weight=min_weight,
        similarity_agg=similarity_agg,
        data_weight_influence=data_weight_influence,
        combine_mode=combine_mode,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )


# ============================================================
# Strategy
# ============================================================
class FedCosinePairwise(FedAvg):
    """FedAvg dengan bobot = f(similarity antar representasi client
    [utama], jumlah data [sekunder])."""

    def __init__(
        self,
        *args,
        temperature: float = 0.1,
        min_weight: float = 0.0,
        similarity_agg: str = "mean",
        data_weight_influence: float = 0.3,
        combine_mode: str = "geometric",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if similarity_agg not in ("mean", "median"):
            raise ValueError("similarity_agg harus 'mean' atau 'median'")
        if not (0.0 <= data_weight_influence <= 1.0):
            raise ValueError("data_weight_influence harus di rentang [0, 1]")
        if combine_mode not in ("linear", "geometric", "similarity_only"):
            raise ValueError(
                "combine_mode harus 'linear', 'geometric', atau 'similarity_only'"
            )

        self.temperature = temperature
        self.min_weight = min_weight
        self.similarity_agg = similarity_agg
        self.data_weight_influence = data_weight_influence
        self.combine_mode = combine_mode

        # Untuk logging/analisis dari luar kelas
        self.last_similarity_matrix: Optional[np.ndarray] = None

    def _representativeness_scores(self, sim_matrix: np.ndarray) -> np.ndarray:
        """Skor tiap client = mean/median similarity ke SEMUA client lain
        (similarity ke diri sendiri diabaikan)."""
        n = sim_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)

        if self.similarity_agg == "mean":
            return (sim_matrix * mask).sum(axis=1) / (n - 1)

        return np.array([np.median(sim_matrix[i][mask[i]]) for i in range(n)])

    def _combine_weights(
        self, sim_w: np.ndarray, data_w: Optional[np.ndarray]
    ) -> np.ndarray:
        """Gabungkan bobot similarity (utama) & data (sekunder).

        Kalau combine_mode="similarity_only", data_w diabaikan total
        (boleh None) dan hasilnya murni sim_w.
        """
        if self.combine_mode == "similarity_only":
            return sim_w / sim_w.sum()

        a, eps = self.data_weight_influence, 1e-12

        if self.combine_mode == "linear":
            w = (1.0 - a) * sim_w + a * data_w
        else:  # geometric
            w = np.power(sim_w + eps, 1.0 - a) * np.power(data_w + eps, a)

        return w / w.sum()

    def aggregate_fit(self, server_round: int, results, failures):
        """Agregasi hasil training client dengan bobot similarity
        (utama) + jumlah data (sekunder)."""

        if not results:
            return None, {}

        client_ndarrays = [parameters_to_ndarrays(r.parameters) for _, r in results]
        num_examples = np.array(
            [r.num_examples for _, r in results], dtype=np.float64
        )

        # 1) similarity antar representasi (parameter), sebelum agregasi apapun
        thetas = _flatten_all(client_ndarrays)
        sim_matrix = _pairwise_cosine_matrix(thetas)
        self.last_similarity_matrix = sim_matrix

        # 2) skor representativitas -> softmax -> sim_weights (FAKTOR UTAMA)
        scores = self._representativeness_scores(sim_matrix)
        t = max(self.temperature, 1e-6)
        scaled = scores / t
        scaled -= scaled.max()
        sim_weights = np.exp(scaled)
        sim_weights /= sim_weights.sum()

        # 3) data_weights -- tetap dihitung untuk keperluan logging/metrics,
        #    tapi TIDAK dipakai untuk bobot akhir kalau combine_mode
        #    "similarity_only" (lihat _combine_weights).
        if num_examples.sum() <= 0:
            data_weights = np.full_like(num_examples, 1.0 / len(num_examples))
        else:
            data_weights = num_examples / num_examples.sum()

        # 4) gabungkan (similarity dominan, atau murni similarity kalau
        #    combine_mode="similarity_only")
        weights = self._combine_weights(sim_weights, data_weights)
        if self.min_weight > 0.0:
            weights = np.maximum(weights, self.min_weight)
            weights /= weights.sum()

        # 5) weighted average per layer
        aggregated = []
        for layer_idx in range(len(client_ndarrays[0])):
            stack = np.stack([c[layer_idx] for c in client_ndarrays], axis=0)
            w = weights.reshape([-1] + [1] * (stack.ndim - 1))
            layer = np.sum(stack * w, axis=0).astype(
                client_ndarrays[0][layer_idx].dtype
            )
            aggregated.append(layer)

        aggregated_parameters = ndarrays_to_parameters(aggregated)

        # 6) metrics
        metrics: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn is not None:
            metrics = self.fit_metrics_aggregation_fn(
                [(r.num_examples, r.metrics) for _, r in results]
            )

        for idx, (_, r) in enumerate(results):
            name = r.metrics.get("client_name", f"client_{idx}")
            metrics[f"pairwise_sim_score_{name}"] = float(scores[idx])
            metrics[f"sim_weight_{name}"] = float(sim_weights[idx])
            metrics[f"data_weight_{name}"] = float(data_weights[idx])
            metrics[f"agg_weight_{name}"] = float(weights[idx])

        return aggregated_parameters, metrics

        
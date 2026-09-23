from typing import Dict, List, Optional, Tuple

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
# Utility Functions
# ============================================================

def _flatten(ndarrays: List[np.ndarray]) -> np.ndarray:
    """Flatten seluruh parameter model menjadi satu vektor."""
    return np.concatenate([np.asarray(param).flatten() for param in ndarrays])


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Menghitung cosine similarity antara dua vektor.

    Returns:
        Nilai cosine similarity dalam range [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < eps or norm_b < eps:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================
# Standard FedAvg Strategy
# ============================================================

def build_fedavg_strategy(
    num_clients: int,
    init_params: Parameters,
    evaluate_fn,
    fit_metrics_agg,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.0,
) -> FedAvg:
    """
    Membuat strategy FedAvg standar Flower.
    """
    return FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )


# ============================================================
# FedCosine Strategy Builder
# ============================================================

def build_fedcosine_strategy(
    num_clients: int,
    init_params: Parameters,
    evaluate_fn,
    fit_metrics_agg,
    temperature: float = 0.1,
    min_weight: float = 0.0,
) -> "FedCosine":
    """
    Membuat strategy FedCosine.

    Args:
        num_clients:
            Jumlah client yang digunakan dalam federated learning.

        init_params:
            Parameter awal model global.

        evaluate_fn:
            Fungsi evaluasi model global.

        fit_metrics_agg:
            Fungsi agregasi metrics dari client.

        temperature:
            Mengontrol ketajaman softmax.
            Nilai kecil -> client dengan similarity tinggi
            mendapat bobot lebih besar.

        min_weight:
            Bobot minimum untuk setiap client.
            Nilai 0 berarti tidak ada batas minimum.
    """
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


# ============================================================
# FedCosine Strategy
# ============================================================

class FedCosine(FedAvg):
    """
    Custom FedAvg dengan pembobotan berdasarkan cosine similarity.

    Berbeda dengan FedAvg standar yang menggunakan `num_examples`
    sebagai dasar bobot agregasi, FedCosine menentukan bobot client
    berdasarkan cosine similarity antara parameter model client dan
    parameter model global pada ronde sebelumnya.

    Mekanisme:

        1. Ambil parameter model dari setiap client.
        2. Hitung cosine similarity terhadap model global sebelumnya.
        3. Ubah similarity menjadi bobot menggunakan softmax.
        4. Lakukan weighted average pada setiap layer.
        5. Gunakan hasil agregasi sebagai model global berikutnya.

    Args:
        temperature:
            Mengontrol seberapa tajam perbedaan bobot antar client.

            - kecil, misalnya 0.05:
              client dengan similarity tinggi lebih dominan.

            - besar, misalnya 1.0:
              bobot antar client menjadi lebih merata.

        min_weight:
            Bobot minimum untuk setiap client.
            Berguna untuk mencegah client tertentu mendapatkan
            bobot terlalu kecil.
    """

    def __init__(
        self,
        *args,
        temperature: float = 0.1,
        min_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temperature = temperature
        self.min_weight = min_weight

        # Parameter global dari ronde sebelumnya.
        self._prev_global_flat: Optional[np.ndarray] = None

    # --------------------------------------------------------
    # Aggregate Fit
    # --------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        """
        Mengagregasi hasil training dari client menggunakan
        cosine similarity sebagai bobot.
        """

        # ----------------------------------------------------
        # 1. Validasi hasil client
        # ----------------------------------------------------

        if not results:
            return None, {}

        # ----------------------------------------------------
        # 2. Ambil parameter dari setiap client
        # ----------------------------------------------------

        client_thetas: List[np.ndarray] = []
        client_ndarrays: List[List[np.ndarray]] = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)

            client_ndarrays.append(ndarrays)
            client_thetas.append(_flatten(ndarrays))

        # ----------------------------------------------------
        # 3. Tentukan reference model
        # ----------------------------------------------------
        #
        # Ronde pertama:
        #   belum ada global model sebelumnya, sehingga
        #   digunakan rata-rata parameter client sebagai
        #   reference sementara.
        #
        # Ronde berikutnya:
        #   gunakan global model dari ronde sebelumnya.
        # ----------------------------------------------------

        if self._prev_global_flat is None:
            reference = np.mean(
                np.stack(client_thetas, axis=0),
                axis=0,
            )
        else:
            reference = self._prev_global_flat

        # ----------------------------------------------------
        # 4. Hitung cosine similarity
        # ----------------------------------------------------

        similarities = np.array(
            [
                cosine_similarity(theta, reference)
                for theta in client_thetas
            ]
        )

        # ----------------------------------------------------
        # 5. Konversi similarity menjadi aggregation weights
        # ----------------------------------------------------

        temperature = max(self.temperature, 1e-6)

        scaled_similarity = similarities / temperature

        # Numerical stability untuk softmax
        scaled_similarity -= scaled_similarity.max()

        weights = np.exp(scaled_similarity)
        weights /= weights.sum()

        # ----------------------------------------------------
        # 6. Terapkan minimum weight jika diperlukan
        # ----------------------------------------------------

        if self.min_weight > 0.0:
            weights = np.maximum(
                weights,
                self.min_weight,
            )

            # Normalisasi ulang agar total bobot = 1
            weights /= weights.sum()

        # ----------------------------------------------------
        # 7. Weighted aggregation setiap layer
        # ----------------------------------------------------

        num_layers = len(client_ndarrays[0])

        aggregated_ndarrays: List[np.ndarray] = []

        for layer_idx in range(num_layers):

            layer_stack = np.stack(
                [
                    client_ndarrays[client_idx][layer_idx]
                    for client_idx in range(len(client_ndarrays))
                ],
                axis=0,
            )

            # Ubah shape weights agar broadcasting sesuai
            weight_shape = [-1] + [1] * (layer_stack.ndim - 1)

            reshaped_weights = weights.reshape(weight_shape)

            aggregated_layer = np.sum(
                layer_stack * reshaped_weights,
                axis=0,
            )

            # Pertahankan dtype parameter client
            aggregated_layer = aggregated_layer.astype(
                client_ndarrays[0][layer_idx].dtype
            )

            aggregated_ndarrays.append(aggregated_layer)

        # ----------------------------------------------------
        # 8. Simpan global model untuk ronde berikutnya
        # ----------------------------------------------------

        self._prev_global_flat = _flatten(
            aggregated_ndarrays
        )

        aggregated_parameters = ndarrays_to_parameters(
            aggregated_ndarrays
        )

        # ----------------------------------------------------
        # 9. Aggregate metrics
        # ----------------------------------------------------

        aggregated_metrics: Dict[str, Scalar] = {}

        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics)
                for _, fit_res in results
            ]

            aggregated_metrics = (
                self.fit_metrics_aggregation_fn(fit_metrics)
            )

        # ----------------------------------------------------
        # 10. Tambahkan cosine similarity & aggregation weight
        # ----------------------------------------------------

        for idx, (_, fit_res) in enumerate(results):

            client_name = fit_res.metrics.get(
                "client_name",
                f"client_{idx}",
            )

            aggregated_metrics[
                f"cos_sim_{client_name}"
            ] = float(similarities[idx])

            aggregated_metrics[
                f"agg_weight_{client_name}"
            ] = float(weights[idx])

        # ----------------------------------------------------
        # 11. Return hasil agregasi
        # ----------------------------------------------------

        return aggregated_parameters, aggregated_metrics

# ============================================================
# FedCosine Pairwise Strategy (dengan visualisasi similarity)
# ============================================================
# Strategy custom untuk Flower (flwr) yang meng-agregasi model
# client berdasarkan PAIRWISE cosine similarity antar client,
# dan menyediakan visualisasi untuk melihat client mana yang
# paling mirip (nearest) dan paling beda (farthest) satu sama
# lain di tiap round.
# ============================================================

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def _flatten(ndarrays: List[np.ndarray]) -> np.ndarray:
    """Gabungkan semua layer parameter jadi satu vector 1D."""
    return np.concatenate([arr.flatten() for arr in ndarrays])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Hitung cosine similarity antara dua vector."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ============================================================
# FedCosinePairwise Strategy
# ============================================================
class FedCosinePairwise(FedAvg):
    """
    Custom FedAvg dengan pembobotan berdasarkan PAIRWISE cosine
    similarity antar client (dihitung sebelum agregasi apapun).

    Mekanisme:

        1. Ambil parameter (atau feature extraction) tiap client.
        2. Hitung cosine similarity ANTAR SEMUA PASANGAN client
           -> matriks similarity N x N.
        3. Skor tiap client = mean/median similarity-nya terhadap
           semua client lain (bukan terhadap reference tunggal).
        4. Ubah skor jadi bobot pakai softmax.
        5. Lakukan weighted average per layer.
        6. Hasil agregasi jadi model global ronde ini.
        7. (Baru) Visualisasikan matriks similarity + client
           terdekat/terjauh tiap client, disimpan sebagai PNG
           dan dicatat ke metrics.

    Args:
        temperature:
            Mengontrol ketajaman softmax.
        min_weight:
            Bobot minimum per client.
        similarity_agg:
            "mean" atau "median", untuk agregasi skor representativitas.
        visualize:
            Jika True, otomatis membuat & menyimpan visualisasi
            similarity di setiap round pada aggregate_fit.
        visualize_show:
            Jika True, panggil plt.show() (biasanya dimatikan saat
            training panjang / headless server).
        visualize_dir:
            Folder untuk menyimpan gambar hasil visualisasi.
    """

    def __init__(
        self,
        *args,
        temperature: float = 0.1,
        min_weight: float = 0.0,
        similarity_agg: str = "mean",
        visualize: bool = True,
        visualize_show: bool = False,
        visualize_dir: str = ".",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temperature = temperature
        self.min_weight = min_weight

        if similarity_agg not in ("mean", "median"):
            raise ValueError("similarity_agg harus 'mean' atau 'median'")
        self.similarity_agg = similarity_agg

        self.visualize = visualize
        self.visualize_show = visualize_show
        self.visualize_dir = visualize_dir

        # Disimpan untuk keperluan logging/analisis dari luar kelas
        self.last_similarity_matrix: Optional[np.ndarray] = None
        self.last_client_names: Optional[List[str]] = None
        self.last_nearest_farthest: Optional[Dict[str, Dict[str, Any]]] = None

        # History antar round, untuk lihat tren "keterasingan" client
        # format: {client_name: [(round, nearest_sim, farthest_sim), ...]}
        self.similarity_history: Dict[str, List[Tuple[int, float, float]]] = {}

    # --------------------------------------------------------
    # Helper: matriks pairwise similarity
    # --------------------------------------------------------

    def _pairwise_similarity_matrix(
        self, thetas: List[np.ndarray]
    ) -> np.ndarray:
        """Hitung matriks cosine similarity N x N antar semua client."""
        n = len(thetas)
        sim_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    s = cosine_similarity(thetas[i], thetas[j])
                    sim_matrix[i, j] = s
                    sim_matrix[j, i] = s

        return sim_matrix

    # --------------------------------------------------------
    # Helper: skor representativitas tiap client
    # --------------------------------------------------------

    def _representativeness_scores(
        self, sim_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Skor tiap client = agregasi (mean/median) similarity-nya
        terhadap SEMUA client lain (similarity ke diri sendiri
        dibuang dulu).
        """
        n = sim_matrix.shape[0]
        scores = np.zeros(n, dtype=np.float64)

        for i in range(n):
            others = np.delete(sim_matrix[i], i)

            if self.similarity_agg == "mean":
                scores[i] = others.mean()
            else:
                scores[i] = np.median(others)

        return scores

    # --------------------------------------------------------
    # Helper: cari client terdekat & terjauh untuk tiap client
    # --------------------------------------------------------

    def _nearest_farthest_per_client(
        self,
        sim_matrix: np.ndarray,
        client_names: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Untuk tiap client, cari:
        - client lain dengan similarity TERTINGGI (paling mirip / terdekat)
        - client lain dengan similarity TERENDAH (paling beda / terjauh)
        """
        n = sim_matrix.shape[0]
        result: Dict[str, Dict[str, Any]] = {}

        for i in range(n):
            row = sim_matrix[i].copy()
            row[i] = -np.inf  # buang diri sendiri biar gak kepilih

            nearest_idx = int(np.argmax(row))
            farthest_idx = int(np.argmin(row))

            result[client_names[i]] = {
                "nearest_client": client_names[nearest_idx],
                "nearest_similarity": float(sim_matrix[i, nearest_idx]),
                "farthest_client": client_names[farthest_idx],
                "farthest_similarity": float(sim_matrix[i, farthest_idx]),
            }

        return result

    # --------------------------------------------------------
    # Visualisasi: heatmap similarity + bar chart nearest/farthest
    # --------------------------------------------------------

    def visualize_similarity(
        self,
        sim_matrix: np.ndarray,
        client_names: List[str],
        server_round: int,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Bikin 2 visualisasi berdampingan:
        1. Heatmap N x N pairwise cosine similarity antar client.
        2. Bar chart nearest vs farthest similarity tiap client,
           dengan nama client tetangga sebagai label.

        Return dict nearest/farthest per client (juga dipakai untuk
        update similarity_history).
        """
        nf = self._nearest_farthest_per_client(sim_matrix, client_names)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- Plot 1: Heatmap similarity matrix ---
        sns.heatmap(
            sim_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=client_names,
            yticklabels=client_names,
            vmin=-1, vmax=1,
            ax=axes[0],
            cbar_kws={"label": "Cosine Similarity"},
        )
        axes[0].set_title(f"Pairwise Similarity Matrix - Round {server_round}")
        axes[0].set_xlabel("Client")
        axes[0].set_ylabel("Client")

        # --- Plot 2: Nearest vs Farthest similarity per client ---
        names = list(nf.keys())
        nearest_vals = [nf[c]["nearest_similarity"] for c in names]
        farthest_vals = [nf[c]["farthest_similarity"] for c in names]

        x = np.arange(len(names))
        width = 0.35

        axes[1].bar(
            x - width / 2, nearest_vals, width,
            label="Nearest (paling mirip)", color="#2ecc71",
        )
        axes[1].bar(
            x + width / 2, farthest_vals, width,
            label="Farthest (paling beda)", color="#e74c3c",
        )

        # anotasi nama client tetangga di atas tiap bar
        for idx, c in enumerate(names):
            axes[1].annotate(
                nf[c]["nearest_client"],
                (x[idx] - width / 2, nearest_vals[idx]),
                ha="center", va="bottom", fontsize=8, rotation=45,
            )
            axes[1].annotate(
                nf[c]["farthest_client"],
                (x[idx] + width / 2, farthest_vals[idx]),
                ha="center", va="bottom", fontsize=8, rotation=45,
            )

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha="right")
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_title(f"Nearest vs Farthest Client - Round {server_round}")
        axes[1].legend()
        axes[1].axhline(0, color="gray", linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return nf

    # --------------------------------------------------------
    # Visualisasi: tren keterasingan client sepanjang round
    # --------------------------------------------------------

    def visualize_history(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot garis nearest_similarity & farthest_similarity tiap
        client dari round ke round. Berguna untuk melihat apakah
        ada client yang konsisten jadi outlier (farthest similarity
        rendah terus-menerus) atau makin lama makin "menyatu"
        dengan client lain.
        """
        if not self.similarity_history:
            print("Belum ada history similarity. Jalankan beberapa round dulu.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

        for client_name, records in self.similarity_history.items():
            rounds = [r for r, _, _ in records]
            nearest = [n for _, n, _ in records]
            farthest = [f for _, _, f in records]

            axes[0].plot(rounds, nearest, marker="o", label=client_name)
            axes[1].plot(rounds, farthest, marker="o", label=client_name)

        axes[0].set_title("Nearest Similarity per Round")
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Cosine Similarity")
        axes[0].legend(fontsize=8)

        axes[1].set_title("Farthest Similarity per Round")
        axes[1].set_xlabel("Round")
        axes[1].legend(fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

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
        pairwise cosine similarity sebagai dasar bobot, sekaligus
        membuat visualisasi similarity antar client.
        """

        # ----------------------------------------------------
        # 1. Validasi hasil client
        # ----------------------------------------------------
        if not results:
            return None, {}

        # ----------------------------------------------------
        # 2. Ambil parameter dari setiap client
        # ----------------------------------------------------
        #
        # Kalau mau pakai feature extraction/embedding alih-alih
        # seluruh parameter model, ganti bagian ini dengan vector
        # embedding yang dikirim client lewat fit_res.metrics.
        # ----------------------------------------------------

        client_thetas: List[np.ndarray] = []
        client_ndarrays: List[List[np.ndarray]] = []
        client_names: List[str] = []

        for idx, (_, fit_res) in enumerate(results):
            ndarrays = parameters_to_ndarrays(fit_res.parameters)

            client_ndarrays.append(ndarrays)
            client_thetas.append(_flatten(ndarrays))
            client_names.append(
                str(fit_res.metrics.get("client_name", f"client_{idx}"))
            )

        # ----------------------------------------------------
        # 3. Hitung pairwise similarity SEBELUM ada agregasi apapun
        # ----------------------------------------------------
        sim_matrix = self._pairwise_similarity_matrix(client_thetas)
        self.last_similarity_matrix = sim_matrix
        self.last_client_names = client_names

        # ----------------------------------------------------
        # 4. Skor representativitas tiap client
        # ----------------------------------------------------
        scores = self._representativeness_scores(sim_matrix)

        # ----------------------------------------------------
        # 5. Konversi skor menjadi aggregation weights (softmax)
        # ----------------------------------------------------

        temperature = max(self.temperature, 1e-6)

        scaled_scores = scores / temperature
        scaled_scores -= scaled_scores.max()

        weights = np.exp(scaled_scores)
        weights /= weights.sum()

        # ----------------------------------------------------
        # 6. Terapkan minimum weight jika diperlukan
        # ----------------------------------------------------

        if self.min_weight > 0.0:
            weights = np.maximum(weights, self.min_weight)
            weights /= weights.sum()

        # ----------------------------------------------------
        # 7. Weighted aggregation setiap layer
        # ----------------------------------------------------
        #
        # Ini baru dilakukan SETELAH bobot dari pairwise
        # similarity didapat -- bukan sebelum, seperti versi lama.
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

            weight_shape = [-1] + [1] * (layer_stack.ndim - 1)
            reshaped_weights = weights.reshape(weight_shape)

            aggregated_layer = np.sum(
                layer_stack * reshaped_weights,
                axis=0,
            )

            aggregated_layer = aggregated_layer.astype(
                client_ndarrays[0][layer_idx].dtype
            )

            aggregated_ndarrays.append(aggregated_layer)

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # ----------------------------------------------------
        # 8. Aggregate metrics dari client
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
        # 9. Catat skor & bobot tiap client untuk monitoring
        # ----------------------------------------------------

        for idx, client_name in enumerate(client_names):
            aggregated_metrics[
                f"pairwise_sim_score_{client_name}"
            ] = float(scores[idx])

            aggregated_metrics[
                f"agg_weight_{client_name}"
            ] = float(weights[idx])

        # ----------------------------------------------------
        # 10. Visualisasi: nearest/farthest client + heatmap
        # ----------------------------------------------------

        if self.visualize:
            save_path = f"{self.visualize_dir}sim_round_{server_round}.png"

            nearest_farthest = self.visualize_similarity(
                sim_matrix,
                client_names,
                server_round=server_round,
                save_path=save_path,
                show=self.visualize_show,
            )
            self.last_nearest_farthest = nearest_farthest

            # simpan info nearest/farthest ke metrics, biar kelihatan
            # di logging/dashboard tanpa perlu buka gambar
            for cname, info in nearest_farthest.items():
                aggregated_metrics[f"nearest_of_{cname}"] = info["nearest_client"]
                aggregated_metrics[f"nearest_sim_of_{cname}"] = info["nearest_similarity"]
                aggregated_metrics[f"farthest_of_{cname}"] = info["farthest_client"]
                aggregated_metrics[f"farthest_sim_of_{cname}"] = info["farthest_similarity"]

                # update history untuk tren antar round
                self.similarity_history.setdefault(cname, []).append(
                    (server_round, info["nearest_similarity"], info["farthest_similarity"])
                )

        # ----------------------------------------------------
        # 11. Return hasil agregasi
        # ----------------------------------------------------

        return aggregated_parameters, aggregated_metrics


# ============================================================
# Builder function
# ============================================================
def build_fedcosine_pairwise_strategy(
    num_clients: int,
    init_params: Parameters,
    evaluate_fn=None,
    fit_metrics_agg=None,
    temperature: float = 0.1,
    min_weight: float = 0.0,
    similarity_agg: str = "mean",
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    visualize: bool = True,
    visualize_show: bool = False,
    visualize_dir: str = ".",
) -> FedCosinePairwise:
    """
    Helper untuk bikin instance FedCosinePairwise dengan konfigurasi
    yang biasa dipakai di server_app Flower.
 
    Args:
        num_clients:
            Jumlah client yang ikut federated learning. Dipakai
            sebagai default untuk min_fit_clients / min_evaluate_clients
            / min_available_clients kalau tidak di-override.
        init_params:
            Parameter awal model global (Parameters dari flwr.common).
        evaluate_fn:
            Fungsi evaluasi server-side (opsional), signature sesuai
            FedAvg.evaluate_fn.
        fit_metrics_agg:
            Fungsi agregasi metrics dari client (fit_metrics_aggregation_fn).
        temperature, min_weight, similarity_agg:
            Diteruskan ke FedCosinePairwise, lihat docstring class-nya.
        fraction_fit, fraction_evaluate:
            Proporsi client yang dipanggil tiap round.
        min_fit_clients, min_evaluate_clients:
            Kalau None, default-nya = num_clients (semua client dipakai).
        visualize, visualize_show, visualize_dir:
            Kontrol visualisasi pairwise similarity, lihat FedCosinePairwise.
 
    Returns:
        Instance FedCosinePairwise siap dipakai di ServerAppComponents/start_server.
    """
 
    if min_fit_clients is None:
        min_fit_clients = num_clients
    if min_evaluate_clients is None:
        min_evaluate_clients = num_clients
 
    strategy = FedCosinePairwise(
        temperature=temperature,
        min_weight=min_weight,
        similarity_agg=similarity_agg,
        visualize=visualize,
        visualize_show=visualize_show,
        visualize_dir=visualize_dir,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )
 
    return strategy
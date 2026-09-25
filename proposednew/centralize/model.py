import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Bagian theta: mengekstrak representasi fitur dari input."""

    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x


class ClassifierHead(nn.Module):
    """Bagian phi: mengklasifikasikan fitur menjadi kelas output."""

    def __init__(self, in_dim=32, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class FFNNModel(nn.Module):
    """
    Model gabungan: adapter (opsional) + theta (FeatureExtractor) + phi (ClassifierHead).

    Jika `n_features_raw` diisi (beda dari `in_dim`), model akan punya
    `adapter: Linear(n_features_raw -> in_dim)` sehingga client dengan
    jumlah fitur mentah berbeda tetap bisa dipetakan ke dimensi bersama
    (common_dim) sebelum masuk ke theta. Jika tidak diisi, adapter jadi
    Identity, jadi forward-nya sama seperti versi tanpa adapter.
    """

    def __init__(self, in_dim, num_classes=2, n_features_raw=None):
        super().__init__()
        if n_features_raw is not None and n_features_raw != in_dim:
            self.adapter = nn.Linear(n_features_raw, in_dim)
        else:
            self.adapter = nn.Identity()

        self.theta = FeatureExtractor(in_dim=in_dim)
        self.phi = ClassifierHead(num_classes=num_classes)

    def forward(self, x):
        x = self.adapter(x)
        features = self.theta(x)
        logits = self.phi(features)
        return logits


def build_model(in_dim, num_classes=2, seed=42, n_features_raw=None):
    """
    Helper untuk membangun model dengan seed tetap agar reproducible.

    Args:
        in_dim (int): dimensi fitur bersama/common (input ke theta).
        num_classes (int): jumlah kelas output.
        seed (int): random seed untuk inisialisasi bobot.
        n_features_raw (int, optional): jumlah fitur mentah sebelum adapter.
            Jika None atau sama dengan in_dim, adapter jadi Identity.

    Returns:
        FFNNModel
    """
    torch.manual_seed(seed)
    model = FFNNModel(in_dim=in_dim, num_classes=num_classes, n_features_raw=n_features_raw)
    return model


if __name__ == "__main__":
    # Tanpa adapter aktif (adapter = Identity)
    example_in_dim = 100
    model = build_model(in_dim=example_in_dim)
    print(model)
    print("adapter:", model.adapter)  # Identity

    # Dengan adapter aktif (fitur mentah 150 -> common_dim 100)
    client_model = build_model(in_dim=example_in_dim, n_features_raw=150)
    print(client_model)
    print("adapter:", client_model.adapter)  # Linear(150 -> 100)

    dummy_x = torch.randn(4, 150)
    out = client_model(dummy_x)
    print("output shape:", out.shape)  # -> torch.Size([4, 2])
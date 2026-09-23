# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %%
import pandas as pd
import numpy as np

DATA_PATH = "/Users/muhammadzuamaalamin/Documents/tesis/dataset/kronodroid.csv"

data_raw = pd.read_csv(DATA_PATH)
data = data_raw.copy()

# Buang kolom index bawaan (kalau ada)
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

print(data.shape)

# Cek missing value
print("Missing value sebelum drop:")
print(data.isna().sum().sum())

data = data.dropna().reset_index(drop=True)

print("Missing value setelah drop:")
print(data.isna().sum().sum())

# Cek nilai unik & distribusi kelas target
print(data['Malware'].unique())
print(data['Malware'].value_counts())

# Pastikan semua fitur numerik
feature_columns = [c for c in data.columns if c != "Malware"]
data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors="coerce")
data = data.dropna().reset_index(drop=True)

X = data.drop(columns=["Malware"]).values.astype(np.float32)
y = data["Malware"].astype(int).values
target_names = ["goodware", "malware"]

print("Ukuran X:", X.shape, "| Ukuran y:", y.shape)
print("Distribusi kelas:", np.bincount(y))

# %%
# ## 2. Split Data: Train, Validation, Test

# %%
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

val_ratio = 0.15 / (1 - 0.15)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=val_ratio,
    random_state=42,
    stratify=y_train_val
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t   = torch.tensor(y_val, dtype=torch.long)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

print("Train:", X_train_t.shape, "| Val:", X_val_t.shape, "| Test:", X_test_t.shape)

# %%
# ## 3. Definisi Model (θ dan φ) — pakai nama layer eksplisit (fc1, fc2)
# supaya gampang diakses untuk analisis bobot nanti

# %%
class FeatureExtractor(nn.Module):
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
    def __init__(self, in_dim=32, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

class DrebinModel(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.theta = FeatureExtractor(in_dim=in_dim)
        self.phi = ClassifierHead(num_classes=num_classes)

    def forward(self, x):
        features = self.theta(x)
        logits = self.phi(features)
        return logits

torch.manual_seed(42)
in_dim = X_train_t.shape[1]
model = DrebinModel(in_dim=in_dim, num_classes=2)
model

# %%
# ## 3. Definisi Model (θ dan φ) — pakai nama layer eksplisit (fc1, fc2)
# supaya gampang diakses untuk analisis bobot nanti

# %%
class FeatureExtractor(nn.Module):
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
    def __init__(self, in_dim=32, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

class DrebinModel(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.theta = FeatureExtractor(in_dim=in_dim)
        self.phi = ClassifierHead(num_classes=num_classes)

    def forward(self, x):
        features = self.theta(x)
        logits = self.phi(features)
        return logits

torch.manual_seed(42)
in_dim = X_train_t.shape[1]
model = DrebinModel(in_dim=in_dim, num_classes=2)
model

# %%
model.eval()
with torch.no_grad():
    features_before = model.theta(X_test_t).clone()

weights_before = {
    name: param.clone().detach()
    for name, param in model.theta.named_parameters()
}

print("=== Fitur SEBELUM training (5 sampel pertama) ===")
print(features_before[:5])

# %%
# ## 5. Training + capture fitur laten tiap beberapa epoch (untuk PCA progres)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 200
capture_every = 20  # ambil snapshot fitur tiap 20 epoch

# simpan snapshot fitur laten: {epoch: features_test}
feature_snapshots = {}

# snapshot awal (epoch 0, sebelum training)
model.eval()
with torch.no_grad():
    feature_snapshots[0] = model.theta(X_test_t).clone()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % capture_every == 0 or (epoch + 1) == epochs:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_acc = (val_outputs.argmax(dim=1) == y_val_t).float().mean()

            # simpan snapshot fitur laten dari test set
            feature_snapshots[epoch + 1] = model.theta(X_test_t).clone()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} "
              f"- Val Loss: {val_loss.item():.4f} - Val Acc: {val_acc.item()*100:.2f}%")

print(f"\nJumlah snapshot tersimpan: {len(feature_snapshots)}")
print("Epoch yang di-capture:", list(feature_snapshots.keys()))

# %%
# ## 6. Evaluasi akhir di Test Set

# %%
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1)
    acc = (preds == y_test_t).float().mean()
    print(f"Akurasi pada test set: {acc.item()*100:.2f}%")

# %%
# ## 7. Ambil fitur & bobot SESUDAH training

# %%
model.eval()
with torch.no_grad():
    features_after = model.theta(X_test_t).clone()

weights_after = {
    name: param.clone().detach()
    for name, param in model.theta.named_parameters()
}

print("=== Fitur SESUDAH training (5 sampel pertama) ===")
print(features_after[:5])

diff = (features_after - features_before).abs().mean()
print(f"\nRata-rata perubahan nilai fitur laten (before vs after): {diff.item():.4f}")

print("\n=== Perubahan bobot (before vs after) ===")
for name in weights_before:
    w_diff = (weights_after[name] - weights_before[name]).abs()
    print(f"{name}: rata-rata perubahan={w_diff.mean():.4f}, "
          f"perubahan maksimum={w_diff.max():.4f}")

# %%
# ## 8. Visualisasi PCA: fitur laten before vs after training

# %%
pca_before = PCA(n_components=2).fit_transform(features_before.numpy())
pca_after  = PCA(n_components=2).fit_transform(features_after.numpy())

colors = ["tab:blue", "tab:red"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for cls in range(2):
    idx = (y_test == cls)
    axes[0].scatter(pca_before[idx, 0], pca_before[idx, 1],
                     label=target_names[cls], alpha=0.6, s=15, c=colors[cls])
    axes[1].scatter(pca_after[idx, 0], pca_after[idx, 1],
                     label=target_names[cls], alpha=0.6, s=15, c=colors[cls])

axes[0].set_title("Fitur θ(x) SEBELUM training\n(bobot random)")
axes[1].set_title("Fitur θ(x) SESUDAH training\n(setelah backpropagation)")
for ax in axes:
    ax.legend()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.tight_layout()
plt.savefig("feature_before_after.png", dpi=120)
plt.show()

# %%
# ## 8b. Visualisasi PCA PER EPOCH (progres representasi laten)

# %%
epoch_list = list(feature_snapshots.keys())
n_snapshots = len(epoch_list)

n_cols = 4
n_rows = int(np.ceil(n_snapshots / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, ep in enumerate(epoch_list):
    feats = feature_snapshots[ep].numpy()
    pca_result = PCA(n_components=2).fit_transform(feats)

    ax = axes[i]
    for cls in range(2):
        idx = (y_test == cls)
        ax.scatter(pca_result[idx, 0], pca_result[idx, 1],
                   label=target_names[cls], alpha=0.6, s=10, c=colors[cls])

    title = "Epoch 0 (awal)" if ep == 0 else f"Epoch {ep}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.tick_params(labelsize=7)

    if i == 0:
        ax.legend(fontsize=8)

for j in range(n_snapshots, len(axes)):
    axes[j].axis("off")

plt.suptitle("Progres Fitur Laten θ(x) via PCA — per Epoch", fontsize=14)
plt.tight_layout()
plt.savefig("pca_progress_per_epoch.png", dpi=120)
plt.show()

# %%
# ## 9. Visualisasi Heatmap bobot fc1 (before vs after vs selisih)

# %%
fc1_before = weights_before['fc1.weight'].numpy()
fc1_after  = weights_after['fc1.weight'].numpy()
fc1_diff   = fc1_after - fc1_before

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im0 = axes[0].imshow(fc1_before, cmap='coolwarm', aspect='auto')
axes[0].set_title("fc1.weight — SEBELUM training")
axes[0].set_xlabel(f"Input features ({in_dim})")
axes[0].set_ylabel("Hidden units (64)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(fc1_after, cmap='coolwarm', aspect='auto')
axes[1].set_title("fc1.weight — SESUDAH training")
axes[1].set_xlabel(f"Input features ({in_dim})")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(fc1_diff, cmap='coolwarm', aspect='auto')
axes[2].set_title("Selisih (after - before)")
axes[2].set_xlabel(f"Input features ({in_dim})")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("weights_before_after.png", dpi=120)
plt.show()

# %%
# ## 10. Distribusi bobot fc2 (before vs after)

# %%
fc2_before = weights_before['fc2.weight'].numpy().flatten()
fc2_after  = weights_after['fc2.weight'].numpy().flatten()

plt.figure(figsize=(8, 5))
plt.hist(fc2_before, bins=30, alpha=0.5, label='Sebelum training')
plt.hist(fc2_after, bins=30, alpha=0.5, label='Sesudah training')
plt.title("Distribusi bobot fc2 (layer kedua θ): Sebelum vs Sesudah")
plt.xlabel("Nilai bobot")
plt.ylabel("Frekuensi")
plt.legend()
plt.savefig("weights_distribution.png", dpi=120)
plt.show()

# %%
model.eval()
with torch.no_grad():
    logits_before = model.phi(features_before)
    probs_before  = F.softmax(logits_before, dim=1)
    preds_before  = logits_before.argmax(dim=1)
    acc_before    = (preds_before == y_test_t).float().mean()

    logits_after = model.phi(features_after)
    probs_after  = F.softmax(logits_after, dim=1)
    preds_after  = logits_after.argmax(dim=1)
    acc_after    = (preds_after == y_test_t).float().mean()

print("=== Output φ SEBELUM training (5 sampel pertama) ===")
print("Probabilitas:\n", probs_before[:5])
print(f"Akurasi (masih random): {acc_before.item()*100:.2f}%\n")

print("=== Output φ SESUDAH training (5 sampel pertama) ===")
print("Probabilitas:\n", probs_after[:5])
print(f"Akurasi: {acc_after.item()*100:.2f}%")

# %%
# ## 12. Distribusi confidence prediksi before vs after

# %%
conf_before = probs_before.max(dim=1).values.numpy()
conf_after  = probs_after.max(dim=1).values.numpy()

plt.figure(figsize=(8, 5))
plt.hist(conf_before, bins=20, alpha=0.5, label='Sebelum training', range=(0, 1))
plt.hist(conf_after, bins=20, alpha=0.5, label='Sesudah training', range=(0, 1))
plt.title("Distribusi confidence prediksi φ (probabilitas kelas terpilih)")
plt.xlabel("Confidence (max softmax probability)")
plt.ylabel("Jumlah sampel")
plt.legend()
plt.savefig("head_confidence_before_after.png", dpi=120)
plt.show()

# %%
n_show = 20

fig, axes = plt.subplots(1, 2, figsize=(10, 6))

im0 = axes[0].imshow(probs_before[:n_show].numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=1)
axes[0].set_title("Probabilitas φ — SEBELUM training")
axes[0].set_xlabel("Kelas")
axes[0].set_ylabel("Sampel test ke-")
axes[0].set_xticks(range(2))
axes[0].set_xticklabels(target_names, rotation=20)
plt.colorbar(im0, ax=axes[0], label="Probabilitas")

im1 = axes[1].imshow(probs_after[:n_show].numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=1)
axes[1].set_title("Probabilitas φ — SESUDAH training")
axes[1].set_xlabel("Kelas")
axes[1].set_xticks(range(2))
axes[1].set_xticklabels(target_names, rotation=20)
plt.colorbar(im1, ax=axes[1], label="Probabilitas")

plt.tight_layout()
plt.savefig("head_probability_heatmap.png", dpi=120)
plt.show()
print("Heatmap probabilitas disimpan di head_probability_heatmap.png")

# %% [markdown]
# ## 14. Ringkasan Perbandingan

# %%
print("\n=== RINGKASAN PERBANDINGAN HEAD (φ) ===")
print(f"{'Metrik':<30}{'Sebelum':<15}{'Sesudah':<15}")
print(f"{'Akurasi':<30}{acc_before.item()*100:<15.2f}{acc_after.item()*100:<15.2f}")
print(f"{'Rata-rata confidence':<30}{conf_before.mean():<15.4f}{conf_after.mean():<15.4f}")

# %%




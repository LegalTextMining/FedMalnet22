def partition_clients(y, n_clients=4, mode="iid", alpha=0.5, seed=42, min_size=10):
    """Return list of index array per client. mode: 'iid' | 'non_iid' (Dirichlet)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    if mode == "iid":
        return np.array_split(rng.permutation(len(y)), n_clients)
    if mode != "non_iid":
        raise ValueError("mode harus 'iid' atau 'non_iid'")

    while True:  # ulangi sampai tiap client punya >= min_size sampel
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            cuts = (np.cumsum(rng.dirichlet([alpha] * n_clients)) * len(idx)).astype(int)[:-1]
            for k, p in enumerate(np.split(idx, cuts)):
                parts[k].extend(p)
        if min(map(len, parts)) >= min_size:
            return [np.array(p) for p in parts]


def make_client_datasets(split, **kw):
    """Bagi train set jadi list client {'X','y'}. kw diteruskan ke partition_clients."""
    y = split["y_train_t"].numpy()
    return [
        {"X": split["X_train_t"][i], "y": split["y_train_t"][i]}
        for i in map(torch.as_tensor, partition_clients(y, **kw))
    ]


def print_client_distribution(clients, title=""):
    print(f"== {title} ==")
    for i, c in enumerate(clients):
        print(f"  Client {i}: n={len(c['y'])} | kelas={torch.bincount(c['y'], minlength=2).tolist()}")
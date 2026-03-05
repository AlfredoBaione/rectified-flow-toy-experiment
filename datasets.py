# toy_datasets.py

import torch
from sklearn.datasets import make_moons, make_circles, make_blobs


def sample_distribution(name, n_samples):

    if name == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=0.05)

    elif name == "circles":
        X, _ = make_circles(n_samples=n_samples, noise=0.05)

    elif name == "blobs":
        X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.8)

    elif name == "gaussian":
        return torch.randn(n_samples, 2)

    else:
        raise ValueError(f"Unknown distribution {name}")

    return torch.tensor(X, dtype=torch.float32)
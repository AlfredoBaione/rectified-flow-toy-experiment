# sample.py

import torch
from datasets import sample_distribution


@torch.no_grad()
def euler_sampling(model, n_samples, dist_A, steps=100, device="cpu"):

    model.eval()
    x = sample_distribution(dist_A, n_samples).to(device)

    dt = 1.0 / steps

    for i in range(steps):
        t = torch.ones(n_samples, device=device) * (i / steps)
        v = model(x, t)
        x = x + v * dt

    return x.cpu()
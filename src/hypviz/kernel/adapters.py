"""Boundary adapters: accept torch / geoopt tensors without depending on them."""
import numpy as np


def as_numpy(x):
    if hasattr(x, "detach"):  # torch.Tensor / geoopt.ManifoldTensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)

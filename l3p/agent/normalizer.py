"""Running mean/std normalizer for observations and goals.

The paper normalizes network inputs by running means and standard deviations
per input dimension (Appendix D, for Fetch tasks in particular). We apply it to
observations and goals across all environments; it is a no-op statistically
until enough data has been seen.
"""

from __future__ import annotations

import numpy as np


class Normalizer:
    def __init__(self, size: int, eps: float = 1e-2, clip_range: float = 5.0):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range
        self.sum = np.zeros(size, dtype=np.float64)
        self.sumsq = np.zeros(size, dtype=np.float64)
        self.count = 1e-4
        self.mean = np.zeros(size, dtype=np.float32)
        self.std = np.ones(size, dtype=np.float32)

    def update(self, x: np.ndarray) -> None:
        x = x.reshape(-1, self.size).astype(np.float64)
        self.sum += x.sum(axis=0)
        self.sumsq += (x ** 2).sum(axis=0)
        self.count += x.shape[0]
        self.mean = (self.sum / self.count).astype(np.float32)
        var = self.sumsq / self.count - (self.sum / self.count) ** 2
        self.std = np.sqrt(np.maximum(self.eps ** 2, var)).astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -self.clip_range, self.clip_range)

    def normalize_torch(self, x):
        """Normalize a torch tensor with the current running mean/std."""
        import torch
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return torch.clamp((x - m) / s, -self.clip_range, self.clip_range)

    def state_dict(self) -> dict:
        return dict(sum=self.sum, sumsq=self.sumsq, count=self.count,
                    mean=self.mean, std=self.std)

    def load_state_dict(self, d: dict) -> None:
        self.sum, self.sumsq, self.count = d["sum"], d["sumsq"], d["count"]
        self.mean, self.std = d["mean"], d["std"]

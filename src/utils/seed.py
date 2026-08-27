"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # These flags make convolution choices reproducible at a possible speed cost.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from __future__ import annotations

import random
from typing import TypeAlias

import numpy as np
import torch

CameraSource: TypeAlias = int | str


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def l2_normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.copy()
    return vec / norm


def parse_camera_source(raw: str) -> CameraSource:
    stripped = raw.strip()
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    return stripped

from __future__ import annotations

import random
import socket
from datetime import datetime, timezone
from typing import TypeAlias

import numpy as np
import torch

CameraSource: TypeAlias = int | str


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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


def get_access_urls(host: str, port: int) -> list[str]:
    if host not in {"0.0.0.0", "::", ""}:
        return [f"http://{host}:{port}"]

    addresses: set[str] = {"127.0.0.1"}
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
        for info in infos:
            addr = info[4][0]
            if addr:
                addresses.add(addr)
    except socket.gaierror:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            addresses.add(sock.getsockname()[0])
    except OSError:
        pass

    preferred = sorted(addr for addr in addresses if not addr.startswith("127."))
    urls = [f"http://{addr}:{port}" for addr in preferred]
    urls.append(f"http://127.0.0.1:{port}")
    return urls

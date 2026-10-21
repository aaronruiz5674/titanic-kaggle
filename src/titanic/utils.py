"""Utilidades transversales: entorno, reproducibilidad y trazabilidad."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env(required: tuple[str, ...] = ()) -> None:
    """Carga `.env` sin sobrescribir variables ya definidas y valida las obligatorias."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise OSError(f"Faltan variables de entorno obligatorias: {missing}")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fija todas las fuentes de aleatoriedad conocidas."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_info() -> dict[str, str]:
    """SHA y estado del árbol de trabajo, para trazar cada run con su código exacto."""

    def _run(*args: str) -> str:
        return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()

    try:
        return {
            "git_sha": _run("git", "rev-parse", "HEAD"),
            "git_dirty": str(bool(_run("git", "status", "--porcelain"))),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": "unknown", "git_dirty": "unknown"}

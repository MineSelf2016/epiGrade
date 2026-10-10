from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

__all__ = ["ExternalMethodError", "available", "run_external"]

CONDA_ROOT = Path(os.environ.get(
    "EPIGRADE_CONDA_ROOT", str(Path.home() / "miniconda3" / "envs")))
WORKERS = Path(__file__).resolve().parents[1] / "external"

REGISTRY = {
    "scipenn": ("epigrade-scipenn", "python", "scipenn_worker.py"),
    "totalvi": ("epigrade-tv-old", "python", "totalvi_worker.py"),
    "seurat_r": ("epigrade-seurat", "Rscript", "seurat_worker.R"),
}


class ExternalMethodError(RuntimeError):
    pass


def interpreter(name: str) -> Path:
    environment, binary, _ = REGISTRY[name]
    override = os.environ.get(f"EPIGRADE_{name.upper()}_BIN")
    return Path(override) if override else CONDA_ROOT / environment / "bin" / binary


def available(name: str) -> bool:
    return name in REGISTRY and interpreter(name).exists()


def run_external(name: str, reference_rna: np.ndarray,
                 reference_protein: np.ndarray, query_rna: np.ndarray,
                 *, extra: dict | None = None, timeout: int = 10800,
                 verbose: bool = False,
                 diagnostics: dict | None = None) -> np.ndarray:
    if name not in REGISTRY:
        raise KeyError(f"{name} is not a registered external method")
    binary = interpreter(name)
    if not binary.exists():
        raise ExternalMethodError(
            f"{name}: {binary} does not exist; build the environment first")
    worker = WORKERS / REGISTRY[name][2]
    if not worker.exists():
        raise ExternalMethodError(f"{name}: worker {worker} is missing")

    arrays = {
        "reference_rna": np.ascontiguousarray(reference_rna, dtype=np.float32),
        "reference_protein": np.ascontiguousarray(reference_protein,
                                                  dtype=np.float32),
        "query_rna": np.ascontiguousarray(query_rna, dtype=np.float32),
    }

    with tempfile.TemporaryDirectory(prefix=f"epigrade-{name}-") as directory:
        workspace = Path(directory)
        result = workspace / "output.npy"
        if REGISTRY[name][1] == "Rscript":
            payload = workspace / "input.json"
            shapes = {}
            for key, value in arrays.items():
                value.tofile(workspace / f"{key}.bin")
                shapes[key] = list(value.shape)
            result = workspace / "output.bin"
            payload.write_text(json.dumps(
                {"shapes": shapes, "dir": str(workspace),
                 "output": str(result),
                 **{k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in (extra or {}).items()}}))
        else:
            payload = workspace / "input.npz"
            np.savez_compressed(
                payload, **arrays,
                **{k: np.asarray(v) for k, v in (extra or {}).items()})

        completed = subprocess.run(
            [str(binary), str(worker), str(payload), str(result)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(workspace))
        if verbose and completed.stdout:
            print(completed.stdout.strip()[-2000:], flush=True)
        if completed.returncode != 0 or not result.exists():
            tail = (completed.stderr or completed.stdout or "").strip()
            raise ExternalMethodError(
                f"{name} failed (exit {completed.returncode}):\n{tail[-2500:]}")

        if result.suffix == ".bin":
            prediction = np.fromfile(result, dtype=np.float32).reshape(
                query_rna.shape[0], reference_protein.shape[1])
        else:
                prediction = np.load(result)
        note = workspace / "diagnostics.json"
        if diagnostics is not None and note.exists():
            diagnostics.update(json.loads(note.read_text()))

    if prediction.shape != (query_rna.shape[0], reference_protein.shape[1]):
        raise ExternalMethodError(
            f"{name} returned {prediction.shape}, expected "
            f"{(query_rna.shape[0], reference_protein.shape[1])}")
    if not np.isfinite(prediction).all():
        raise ExternalMethodError(f"{name} returned non-finite predictions")
    return np.asarray(prediction, dtype=np.float32)

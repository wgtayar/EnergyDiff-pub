from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import torch


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    config_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    samples_dir: Path
    eval_dir: Path
    plots_dir: Path
    manifests_dir: Path
    progressive_dir: Path


def resolve_artifact_paths(run_root: str | Path) -> ArtifactPaths:
    root = Path(run_root).expanduser()
    return ArtifactPaths(
        root=root,
        config_dir=root / "config",
        logs_dir=root / "logs",
        checkpoints_dir=root / "checkpoints",
        samples_dir=root / "samples",
        eval_dir=root / "eval",
        plots_dir=root / "plots",
        manifests_dir=root / "manifests",
        progressive_dir=root / "progressive",
    )


def ensure_artifact_dirs(run_root: str | Path) -> ArtifactPaths:
    paths = resolve_artifact_paths(run_root)
    for directory in (
        paths.root,
        paths.config_dir,
        paths.logs_dir,
        paths.checkpoints_dir,
        paths.samples_dir,
        paths.eval_dir,
        paths.plots_dir,
        paths.manifests_dir,
        paths.progressive_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def configure_root_logger(log_path: str | Path | None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_device_summary(
    logger: logging.Logger,
    *,
    amp_enabled: bool,
    mixed_precision_type: str,
) -> dict[str, str | bool | int]:
    cuda_available = torch.cuda.is_available()
    chosen_device = "cuda" if cuda_available else "cpu"
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_available and gpu_count > 0 else "N/A"
    precision_mode = mixed_precision_type if amp_enabled else "fp32"

    summary = {
        "chosen_device": chosen_device,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "precision_mode": precision_mode,
    }

    logger.info("Runtime device summary:")
    logger.info("  chosen_device=%s", chosen_device)
    logger.info("  cuda_available=%s", cuda_available)
    logger.info("  gpu_count=%s", gpu_count)
    logger.info("  gpu_name=%s", gpu_name)
    logger.info("  precision_mode=%s", precision_mode)

    return summary

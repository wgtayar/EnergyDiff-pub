from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from functools import partial
import json
import logging
from multiprocessing import cpu_count
import os
from pathlib import Path
import shutil
import sys

import torch
import pandas as pd
try:
    import wandb
except ImportError:
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_SRC_DIR = REPO_ROOT / "src" / "data"
if str(DATA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SRC_DIR))

from daily_profile_metrics import compact_metrics, evaluate_daily_profile_bundle, flatten_sample_tensor, load_daily_csv, maybe_denormalize, resolve_scaling_context
from experiment_layout import append_records, ensure_run_layout, merge_run_manifest, write_run_manifest
from industrial_scorecard import build_industrial_checkpoint_scorecard

from energydiff.dataset import NAME_SEASONS, PIT, standard_normal_cdf, standard_normal_icdf
from energydiff.diffusion import Trainer1D
from energydiff.diffusion.dataset import ConditionalDataset1D
from energydiff.diffusion.dpm_solver import DPMSolverSampler
from energydiff.utils import generate_time_id
from energydiff.utils.argument_parser import argument_parser, save_config
from energydiff.utils.eval import (
    MkMMD,
    UMAPEvalCollection,
    calculate_frechet,
    get_mapper_label,
    kl_divergence,
    ks_test_d,
    source_mean,
    source_std,
    target_mean,
    target_std,
    ws_distance,
)
from energydiff.utils.initializer import (
    create_backbone,
    create_cond_embedder_wrapped,
    create_dataset,
    create_diffusion,
    create_rectified_flow,
    get_generated_filename,
    get_task_profile_condition,
)
from energydiff.utils.runtime import configure_root_logger, ensure_artifact_dirs, log_device_summary


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def default_experiment_slug(config) -> str:
    family = config.family_filter or (config.dataset_key.split("_", 1)[0] if config.dataset_key else "pooled")
    conditioning = "cond" if config.model.conditioning else "uncond"
    return f"{family}_{conditioning}_daily_{config.data.resolution}_{config.model.model_class}"


def absolutize_config_paths(config) -> None:
    if config.data.root is not None:
        config.data.root = str(Path(config.data.root).expanduser().resolve())
    if config.artifact_root is not None:
        config.artifact_root = str(Path(config.artifact_root).expanduser().resolve())
    if config.run_root is not None:
        config.run_root = str(Path(config.run_root).expanduser().resolve())


def load_dataset_export_manifest(dataset_root: Path) -> dict | None:
    manifest_path = dataset_root / "manifests" / "dataset_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_run_config_summary(config) -> dict:
    return {
        "dataset": {
            "dataset": config.data.dataset,
            "dataset_key": config.dataset_key,
            "family_filter": config.family_filter,
            "resolution": config.data.resolution,
            "normalize": config.data.normalize,
            "pit": config.data.pit,
            "vectorize": config.data.vectorize,
            "style_vectorize": config.data.style_vectorize,
            "vectorize_window_size": config.data.vectorize_window_size,
            "lcl_use_fraction": config.data.lcl_use_fraction,
            "train_season": config.data.train_season,
            "val_season": config.data.val_season,
        },
        "model": {
            "model_class": config.model.model_class,
            "conditioning": config.model.conditioning,
            "dim_base": config.model.dim_base,
            "dropout": config.model.dropout,
            "num_attn_head": config.model.num_attn_head,
            "num_encoder_layer": getattr(config.model, "num_encoder_layer", None),
            "num_decoder_layer": getattr(config.model, "num_decoder_layer", None),
            "dim_feedforward": config.model.dim_feedforward,
            "learn_variance": config.model.learn_variance,
        },
        "training": {
            "num_train_step": config.train.num_train_step,
            "train_batch_size": config.train.batch_size,
            "val_batch_size": config.train.val_batch_size,
            "train_lr": config.train.lr,
            "save_and_sample_every": config.train.save_and_sample_every,
            "val_every": config.train.val_every,
            "heavy_eval_every": config.train.heavy_eval_every,
            "diagnostic_test_metrics": config.train.diagnostic_test_metrics,
            "save_best_checkpoint": config.train.save_best_checkpoint,
            "best_checkpoint_metric": config.train.best_checkpoint_metric,
            "best_checkpoint_filename": config.train.best_checkpoint_filename,
            "best_checkpoint_meta_filename": config.train.best_checkpoint_meta_filename,
        },
        "sampling": {
            "num_sample": config.sample.num_sample,
            "num_sampling_step": config.sample.num_sampling_step,
            "dpm_solver_sample": config.sample.dpm_solver_sample,
        },
    }


def build_run_manifest(config, run_id: str, run_layout, log_path: Path) -> dict:
    best_checkpoint_path = None
    best_checkpoint_meta_path = None
    if config.train.save_best_checkpoint:
        best_checkpoint_path = str(run_layout.checkpoints_dir / config.train.best_checkpoint_filename)
        best_checkpoint_meta_path = str(run_layout.checkpoints_dir / config.train.best_checkpoint_meta_filename)
    return {
        "schema_version": 2,
        "run_id": run_id,
        "experiment_slug": config.experiment_slug,
        "dataset_key": config.dataset_key,
        "family_filter": config.family_filter,
        "dataset_root": str(Path(config.data.root).resolve()),
        "artifact_root": str(Path(config.artifact_root).resolve()),
        "run_root": str(run_layout.root),
        "status": {
            "training": "running",
            "evaluation": "pending",
        },
        "config_summary": build_run_config_summary(config),
        "instrumentation": {
            "light_eval_every": config.train.val_every,
            "heavy_eval_every": config.train.heavy_eval_every,
            "save_and_sample_every": config.train.save_and_sample_every,
            "diagnostic_test_metrics": config.train.diagnostic_test_metrics,
            "save_plots_on_heavy_eval": config.train.save_plots_on_heavy_eval,
            "save_best_checkpoint": config.train.save_best_checkpoint,
            "best_checkpoint_metric": config.train.best_checkpoint_metric,
        },
        "paths": {
            "config_dir": str(run_layout.config_dir),
            "logs_dir": str(run_layout.logs_dir),
            "checkpoints_dir": str(run_layout.checkpoints_dir),
            "samples_dir": str(run_layout.samples_dir),
            "eval_dir": str(run_layout.eval_dir),
            "plots_dir": str(run_layout.plots_dir),
            "manifests_dir": str(run_layout.manifests_dir),
            "train_log": str(log_path),
            "checkpoint_metrics_csv": str(run_layout.manifests_dir / "checkpoint_metrics.csv"),
            "checkpoint_local_metrics_csv": str(run_layout.manifests_dir / "checkpoint_local_metrics.csv"),
            "checkpoint_artifacts_csv": str(run_layout.manifests_dir / "checkpoint_artifacts.csv"),
            "best_checkpoint_path": best_checkpoint_path,
            "best_checkpoint_meta_path": best_checkpoint_meta_path,
        },
        "artifacts": {
            "final_sample": None,
            "latest_checkpoint": None,
            "latest_checkpoint_sample": None,
            "best_checkpoint": {
                "path": best_checkpoint_path,
                "meta_path": best_checkpoint_meta_path,
                "criterion": config.train.best_checkpoint_metric if config.train.save_best_checkpoint else None,
                "selected_step": None,
                "selected_checkpoint_path": None,
            },
        },
        "dataset": {},
        "model": {},
    }


def _stringify_path(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _json_safe_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _json_safe_mapping(record: dict) -> dict:
    return {str(key): _json_safe_value(value) for key, value in record.items()}


def select_best_checkpoint_candidate(
    checkpoint_metrics_path: Path,
    *,
    best_checkpoint_metric: str,
) -> tuple[dict | None, dict]:
    if not checkpoint_metrics_path.exists():
        return None, {}

    checkpoint_metrics = pd.read_csv(checkpoint_metrics_path)
    if checkpoint_metrics.empty:
        return None, {}

    if best_checkpoint_metric == "validation_loss":
        eligible = checkpoint_metrics.loc[
            checkpoint_metrics["validation_loss"].notna()
            & checkpoint_metrics["checkpoint_path"].map(lambda value: _stringify_path(value) != "")
        ].copy()
        if eligible.empty:
            return None, {}
        eligible = eligible.sort_values(["validation_loss", "step"], ascending=[True, True])
        row = eligible.iloc[0].to_dict()
        return (
            {
                "row": row,
                "source_checkpoint_path": _stringify_path(row.get("checkpoint_path")),
                "source_sample_path": _stringify_path(row.get("sample_path")),
                "selection_value": float(row["validation_loss"]),
                "selection_value_key": "validation_loss",
                "selection_label": "validation loss",
                "selection_signal": "validation loss on saved checkpoints",
                "selection_mode": "validation_led",
                "diagnostic_only": False,
                "metrics": _json_safe_mapping(row),
            },
            {},
        )

    if best_checkpoint_metric == "industrial_val_score":
        checkpoint_scorecard = build_industrial_checkpoint_scorecard(checkpoint_metrics)
        if checkpoint_scorecard.empty:
            return None, {}
        eligible = checkpoint_scorecard.loc[
            checkpoint_scorecard["val_industrial_rank_score"].notna()
            & checkpoint_scorecard["checkpoint_path"].map(lambda value: _stringify_path(value) != "")
        ].copy()
        if eligible.empty:
            return None, {}
        eligible = eligible.sort_values(["val_industrial_rank_score", "step"], ascending=[True, True])
        row = eligible.iloc[0].to_dict()
        return (
            {
                "row": row,
                "source_checkpoint_path": _stringify_path(row.get("checkpoint_path")),
                "source_sample_path": _stringify_path(row.get("sample_path")),
                "selection_value": float(row["val_industrial_rank_score"]),
                "selection_value_key": "val_industrial_rank_score",
                "selection_label": "validation industrial weighted rank score",
                "selection_signal": "validation industrial weighted rank score on saved heavy checkpoints",
                "selection_mode": "validation_led",
                "diagnostic_only": False,
                "metrics": _json_safe_mapping(row),
            },
            {"checkpoint_scorecard": checkpoint_scorecard},
        )

    raise ValueError(f"Unsupported best_checkpoint_metric={best_checkpoint_metric}")


def maybe_update_best_checkpoint_artifact(
    *,
    run_id: str,
    run_layout,
    config,
    logger: logging.Logger,
    checkpoint_metrics_path: Path,
) -> dict | None:
    if not config.train.save_best_checkpoint:
        return None

    selected, diagnostics = select_best_checkpoint_candidate(
        checkpoint_metrics_path,
        best_checkpoint_metric=config.train.best_checkpoint_metric,
    )
    if selected is None:
        logger.info(
            "Best-checkpoint selection skipped: criterion=%s has no eligible saved checkpoint yet.",
            config.train.best_checkpoint_metric,
        )
        return None

    source_checkpoint_path = Path(selected["source_checkpoint_path"])
    if not source_checkpoint_path.exists():
        logger.warning("Best-checkpoint source path does not exist yet: %s", source_checkpoint_path)
        return None

    best_checkpoint_path = run_layout.checkpoints_dir / config.train.best_checkpoint_filename
    best_checkpoint_meta_path = run_layout.checkpoints_dir / config.train.best_checkpoint_meta_filename
    existing_meta = {}
    if best_checkpoint_meta_path.exists():
        try:
            existing_meta = json.loads(best_checkpoint_meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = {}

    selected_step = int(_json_safe_value(selected["row"].get("step")))
    selected_checkpoint_path = str(source_checkpoint_path)
    if (
        existing_meta.get("selected_step") == selected_step
        and existing_meta.get("selected_checkpoint_path") == selected_checkpoint_path
        and best_checkpoint_path.exists()
    ):
        return existing_meta

    shutil.copy2(source_checkpoint_path, best_checkpoint_path)
    metadata = {
        "schema_version": 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "experiment_slug": config.experiment_slug,
        "dataset_key": config.dataset_key,
        "family_filter": config.family_filter,
        "selected_step": selected_step,
        "selected_milestone": _json_safe_value(selected["row"].get("milestone")),
        "selected_checkpoint_path": selected_checkpoint_path,
        "selected_sample_path": selected["source_sample_path"] or None,
        "saved_artifact_path": str(best_checkpoint_path),
        "selection_criterion": selected["selection_value_key"],
        "selection_label": selected["selection_label"],
        "selection_signal": selected["selection_signal"],
        "selection_value": float(selected["selection_value"]),
        "validation_led": True,
        "diagnostic_only": bool(selected["diagnostic_only"]),
        "metric_values": selected["metrics"],
    }
    if "checkpoint_scorecard" in diagnostics:
        metadata["selection_reference"] = {
            "checkpoint_metrics_csv": str(checkpoint_metrics_path),
            "criterion": config.train.best_checkpoint_metric,
        }
    best_checkpoint_meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    merge_run_manifest(
        run_layout.root,
        {
            "artifacts": {
                "best_checkpoint": {
                    "path": str(best_checkpoint_path),
                    "meta_path": str(best_checkpoint_meta_path),
                    "criterion": config.train.best_checkpoint_metric,
                    "selected_step": selected_step,
                    "selected_checkpoint_path": selected_checkpoint_path,
                }
            },
            "selection": {
                "best_training_checkpoint": {
                    "criterion": config.train.best_checkpoint_metric,
                    "selection_signal": selected["selection_signal"],
                    "selection_value": float(selected["selection_value"]),
                    "selected_step": selected_step,
                    "selected_checkpoint_path": selected_checkpoint_path,
                    "validation_led": True,
                    "diagnostic_only": bool(selected["diagnostic_only"]),
                }
            },
        },
    )
    logger.info(
        "Best checkpoint updated: criterion=%s step=%s source=%s target=%s",
        config.train.best_checkpoint_metric,
        selected_step,
        source_checkpoint_path,
        best_checkpoint_path,
    )
    return metadata


def build_checkpoint_callback(
    *,
    run_id: str,
    run_layout,
    config,
    checkpoint_scaling_factor,
    logger: logging.Logger,
    reference_df,
    reference_profiles,
    val_df,
    val_profiles,
    test_df,
    test_profiles,
):
    checkpoint_metrics_path = run_layout.manifests_dir / "checkpoint_metrics.csv"
    checkpoint_local_metrics_path = run_layout.manifests_dir / "checkpoint_local_metrics.csv"
    checkpoint_artifacts_path = run_layout.manifests_dir / "checkpoint_artifacts.csv"

    def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
        return {f"{prefix}{key}": float(value) for key, value in metrics.items()}

    def _local_rows(step: int, split_label: str, stats_df):
        rows = []
        for _, row in stats_df.iterrows():
            rows.append(
                {
                    "run_id": run_id,
                    "step": step,
                    "split": split_label,
                    "slot": row["slot"],
                    "real_mean_w": float(row["real_mean_w"]),
                    "synthetic_mean_w": float(row["synthetic_mean_w"]),
                    "mean_abs_diff_w": float(row["mean_abs_diff_w"]),
                    "real_std_w": float(row["real_std_w"]),
                    "synthetic_std_w": float(row["synthetic_std_w"]),
                    "std_abs_diff_w": float(row["std_abs_diff_w"]),
                }
            )
        return rows

    def _callback(payload: dict) -> None:
        step = int(payload["step"])
        event_type = str(payload["event_type"])
        synthetic_profiles = maybe_denormalize(
            flatten_sample_tensor(payload["generated_sample"]),
            config,
            scaling_factor=checkpoint_scaling_factor,
        )
        save_plots = bool(config.train.save_plots_on_heavy_eval and event_type == "heavy")
        checkpoint_plot_dir = run_layout.plots_dir / "checkpoints" / f"step_{step:06d}" if save_plots else None

        val_bundle = evaluate_daily_profile_bundle(
            reference_df=reference_df,
            reference_profiles=reference_profiles,
            real_profiles=val_profiles,
            synthetic_profiles=synthetic_profiles,
            plots_dir=checkpoint_plot_dir / "val" if checkpoint_plot_dir is not None else None,
            max_embedding_points=1000,
            plot_prefix="val",
        )

        metrics_row = {
            "run_id": run_id,
            "step": step,
            "event_type": event_type,
            "milestone": payload["milestone"],
            "train_loss": payload["train_loss"],
            "train_mse": payload["train_mse"],
            "train_vb": payload["train_vb"],
            "validation_loss": payload["validation_loss"],
            "checkpoint_path": payload["checkpoint_path"],
            "sample_path": payload["sample_path"],
            **_prefix_metrics("val_", val_bundle.metrics if event_type == "heavy" else compact_metrics(val_bundle.metrics)),
        }
        legacy_metrics = payload.get("validation_metrics", {}) or {}
        metrics_row.update({f"legacy_val_{key}": float(value) for key, value in legacy_metrics.items()})
        local_rows = []
        if event_type == "heavy":
            local_rows = _local_rows(step, "val", val_bundle.per_timestep_stats)
            if config.train.diagnostic_test_metrics:
                test_bundle = evaluate_daily_profile_bundle(
                    reference_df=reference_df,
                    reference_profiles=reference_profiles,
                    real_profiles=test_profiles,
                    synthetic_profiles=synthetic_profiles,
                    plots_dir=checkpoint_plot_dir / "test_diagnostic" if checkpoint_plot_dir is not None else None,
                    max_embedding_points=1000,
                    plot_prefix="test_diagnostic",
                )
                metrics_row.update(_prefix_metrics("test_", test_bundle.metrics))
                local_rows.extend(_local_rows(step, "test_diagnostic", test_bundle.per_timestep_stats))

        append_records(checkpoint_metrics_path, [metrics_row])
        append_records(
            checkpoint_artifacts_path,
            [
                {
                    "run_id": run_id,
                    "step": step,
                    "event_type": event_type,
                    "milestone": payload["milestone"],
                    "checkpoint_path": payload["checkpoint_path"],
                    "sample_path": payload["sample_path"],
                    "plots_dir": str(checkpoint_plot_dir) if checkpoint_plot_dir is not None else "",
                }
            ],
        )

        if local_rows:
            append_records(checkpoint_local_metrics_path, local_rows)

        best_checkpoint_metadata = maybe_update_best_checkpoint_artifact(
            run_id=run_id,
            run_layout=run_layout,
            config=config,
            logger=logger,
            checkpoint_metrics_path=checkpoint_metrics_path,
        )

        if payload["checkpoint_path"] or payload["sample_path"]:
            merge_run_manifest(
                run_layout.root,
                {
                    "artifacts": {
                        "latest_checkpoint": payload["checkpoint_path"] or None,
                        "latest_checkpoint_sample": payload["sample_path"] or None,
                    }
                },
            )

        if best_checkpoint_metadata is not None:
            logger.info(
                "Best validation checkpoint artifact: step=%s criterion=%s path=%s",
                best_checkpoint_metadata.get("selected_step"),
                best_checkpoint_metadata.get("selection_criterion"),
                best_checkpoint_metadata.get("saved_artifact_path"),
            )

        logger.info(
            "Checkpoint metrics recorded at step=%s event_type=%s checkpoint=%s sample=%s",
            step,
            event_type,
            payload["checkpoint_path"] or "-",
            payload["sample_path"] or "-",
        )

    return _callback


def main():
    config = argument_parser()
    absolutize_config_paths(config)
    exp_id = config.exp_id
    generated_time_id = config.time_id is None
    time_id = config.time_id or generate_time_id()

    config.experiment_slug = config.experiment_slug or default_experiment_slug(config)
    config.dataset_key = config.dataset_key or Path(config.data.root).name
    config.family_filter = config.family_filter or config.dataset_key.split("_", 1)[0]

    if config.train.heavy_eval_every is None:
        config.train.heavy_eval_every = config.train.val_every * 2

    experiments_root = Path(config.artifact_root).expanduser()
    if config.run_root:
        resolved_run_root = Path(config.run_root).expanduser()
    else:
        candidate_run_root = experiments_root / config.experiment_slug / time_id
        if generated_time_id and candidate_run_root.exists():
            suffix = 1
            while True:
                candidate_time_id = f"{time_id}-{suffix:02d}"
                candidate_run_root = experiments_root / config.experiment_slug / candidate_time_id
                if not candidate_run_root.exists():
                    time_id = candidate_time_id
                    break
                suffix += 1
        resolved_run_root = candidate_run_root
    run_layout = ensure_run_layout(resolved_run_root)
    artifact_paths = ensure_artifact_dirs(run_layout.root)
    config.run_root = str(run_layout.root)
    config.time_id = time_id

    train_season = config.data.train_season
    val_season = config.data.val_season
    conditioning = config.model.conditioning
    diffusion_objective = config.diffusion.prediction_type
    log_wandb = config.log_wandb
    if log_wandb and wandb is None:
        logging.getLogger(__name__).warning(
            "wandb logging requested but wandb is not installed; disabling wandb logging."
        )
        log_wandb = False
        config.log_wandb = False

    if conditioning:
        run_id = f"train-diffusion-{config.diffusion.prediction_type}-cond-{time_id}"
    else:
        run_id = f"train-diffusion-{config.diffusion.prediction_type}-{train_season}-{time_id}"

    is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
    log_path = artifact_paths.logs_dir / f"{run_id}.log"
    logger = configure_root_logger(log_path if is_main_process else None)
    logger.info("Experiment starts: %s", run_id)
    logger.info("Experiment slug: %s", config.experiment_slug)
    logger.info("Experiment id: %s", exp_id)
    logger.info("Run root: %s", artifact_paths.root)
    logger.info("Dataset root: %s", Path(config.data.root).resolve())
    logger.info(
        "Config flags: dataset=%s normalize=%s pit=%s vectorize=%s conditioning=%s lcl_use_fraction=%s heavy_eval_every=%s",
        config.data.dataset,
        config.data.normalize,
        config.data.pit,
        config.data.vectorize,
        config.model.conditioning,
        config.data.lcl_use_fraction,
        config.train.heavy_eval_every,
    )

    log_device_summary(
        logger,
        amp_enabled=config.train.amp,
        mixed_precision_type=config.train.mixed_precision_type,
    )

    run_manifest = build_run_manifest(config, time_id, run_layout, log_path)
    if is_main_process:
        write_run_manifest(run_layout.root, run_manifest)

    dataset_root = Path(config.data.root).resolve()
    dataset_export_manifest = load_dataset_export_manifest(dataset_root)
    if dataset_export_manifest is not None:
        logger.info("Dataset export manifest: %s", dataset_root / "manifests" / "dataset_manifest.json")
        if dataset_export_manifest.get("export_fingerprint") is not None:
            logger.info("Dataset export fingerprint: %s", dataset_export_manifest["export_fingerprint"])
        if dataset_export_manifest.get("scaling") is not None:
            logger.info("Dataset export scaling: %s", dataset_export_manifest["scaling"])
    reference_df, reference_profiles = load_daily_csv(dataset_root / "raw" / "lcl_electricity_train.csv")
    val_df, val_profiles = load_daily_csv(dataset_root / "raw" / "lcl_electricity_val.csv")
    test_df, test_profiles = load_daily_csv(dataset_root / "raw" / "lcl_electricity_test.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_collection = create_dataset(config.data)
    pre_transforms = {}
    post_transforms = {}
    pit: PIT = dataset_collection.dataset.pit
    if pit is not None:
        pre_transforms["pit"] = pit.transform
        pre_transforms["erf"] = standard_normal_icdf
        post_transforms["erf"] = standard_normal_cdf
        post_transforms["pit"] = pit.inverse_transform
        _data_config = deepcopy(config.data)
        _data_config.pit = False
        dataset_collection = create_dataset(_data_config)

    all_profile, all_condition = get_task_profile_condition(
        dataset_collection,
        season=train_season,
        conditioning=conditioning,
        lcl_use_fraction=config.data.lcl_use_fraction,
    )

    trainset = ConditionalDataset1D(
        tensor=all_profile["train"],
        condition=all_condition["train"],
        transforms=list(pre_transforms.values()),
    )
    logger.info("Train dataset prepared: %s", trainset)
    logger.info(
        "Split sizes: train=%s val=%s test=%s",
        int(all_profile["train"].shape[0]),
        int(all_profile["val"].shape[0]),
        int(all_profile["test"].shape[0]),
    )
    valset = ConditionalDataset1D(
        tensor=all_profile["val"],
        condition=all_condition["val"],
    )

    cond_embedder = None
    sample_model_kwargs = {}
    if conditioning:
        cond_embedder = create_cond_embedder_wrapped(
            dataset_collection=dataset_collection,
            dim_embedding=config.model.dim_base,
        )
        _sample_cond = [float(NAME_SEASONS.index(val_season))]
        if config.data.dataset == "cossmic" and "area" in config.data.target_labels:
            _sample_cond.append(float(dataset_collection.condition_mapping["area"][config.data.val_area]))
        sample_model_kwargs = {
            "c": torch.tensor(_sample_cond, dtype=torch.float32).reshape(1, len(trainset.list_dim_cond), 1).to(device),
            "cfg_scale": config.train.val_sample_config.cfg_scale,
        }
        _profile, _ = get_task_profile_condition(
            dataset_collection,
            season=val_season,
            conditioning=False,
            area=config.data.val_area,
        )
        valset = ConditionalDataset1D(
            tensor=_profile["val"],
            condition=None,
        )

    num_channel = trainset.num_channel
    seq_length = trainset.sequence_length

    backbone_model = create_backbone(
        config.model,
        num_in_channel=num_channel,
        cond_embedder=cond_embedder,
        seq_length=seq_length,
    ).to(device)
    backbone_model.compile()

    if config.model.resume and config.model.freeze_layers:
        backbone_model.freeze_layers()

    if not config.diffusion.use_rectified_flow:
        create_diffusion_base = partial(
            create_diffusion,
            base_model=backbone_model,
            seq_length=seq_length,
            ddpm_config=config.diffusion,
        )
        full_diffusion = create_diffusion_base(num_sampling_timestep=config.diffusion.num_diffusion_step).to(device)
        spaced_diffusion_model = create_diffusion_base(
            num_sampling_timestep=config.train.val_sample_config.num_sampling_step,
        ).to(device)
    else:
        full_diffusion = create_rectified_flow(
            base_model=backbone_model,
            seq_length=seq_length,
            rf_config=config.diffusion,
            num_discretization_step=config.train.val_sample_config.num_sampling_step,
        ).to(device)
        spaced_diffusion_model = full_diffusion

    mkmmd = MkMMD(kernel_type="rbf", num_kernel=1, kernel_mul=2.0, coefficient="auto")
    umap_eval = UMAPEvalCollection(full_dataset_name=get_mapper_label(config.data))

    def pre_val_fn(x):
        x = dataset_collection.inverse_vectorize_fn(x)
        x = x[:, 0, :]
        return x

    dict_eval_fn = {
        "MkMMD": mkmmd,
        "DirectFD": calculate_frechet,
        "source_mean": source_mean,
        "source_std": source_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "kl_divergence": kl_divergence,
        "ws_distance": ws_distance,
        "ks_test_d": ks_test_d,
    }
    for metric_name, metric_fn in umap_eval.generate_eval_sequence():
        dict_eval_fn[metric_name] = metric_fn

    checkpoint_callback = build_checkpoint_callback(
        run_id=time_id,
        run_layout=run_layout,
        config=config,
        checkpoint_scaling_factor=resolve_scaling_context(
            config=config,
            dataset_manifest=dataset_export_manifest,
            dataset_root=dataset_root,
        )["selected_scaling_factor"],
        logger=logger,
        reference_df=reference_df,
        reference_profiles=reference_profiles,
        val_df=val_df,
        val_profiles=val_profiles,
        test_df=test_df,
        test_profiles=test_profiles,
    )

    trainer = Trainer1D.from_config(
        config.train,
        full_diffusion,
        spaced_diffusion_model,
        dataset=trainset,
        val_dataset=valset,
        num_dataloader_workers=int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", cpu_count())),
        max_val_batch=2,
        sample_model_kwargs=sample_model_kwargs,
        post_transforms=post_transforms.values(),
        pre_eval_fn=pre_val_fn,
        dict_eval_fn=dict_eval_fn,
        log_wandb=config.log_wandb,
        log_id=run_id,
        result_folder=str(artifact_paths.checkpoints_dir),
        checkpoint_callback=checkpoint_callback,
    )

    config.model.num_parameter = count_parameters(backbone_model)
    logger.info("Model parameters: %s", config.model.num_parameter)
    try:
        config.data.scaling_factor = list(map(lambda x: x.item(), dataset_collection.dataset.scaling_factor))
        logger.info("Scaling factor: %s", config.data.scaling_factor)
    except AttributeError:
        logger.info("Scaling factor: not available.")

    if trainer.accelerator.is_main_process:
        if log_wandb and wandb is not None:
            wandb.init(
                project={
                    "wpuq": "HeatDDPM",
                    "wpuq_trafo": "WPuQTrafoDDPm",
                    "wpuq_pv": "WPuQPVDDPM",
                    "lcl_electricity": "LCLDDPM",
                    "cossmic": "CoSSMicDDPM",
                }[config.data.dataset],
                name=run_id,
                config=config.to_dict(),
            )
    else:
        logging.getLogger().setLevel(logging.CRITICAL + 1)

    if is_main_process:
        dataset_patch = {
            "manifest_path": str(dataset_root / "manifests" / "dataset_manifest.json")
            if (dataset_root / "manifests" / "dataset_manifest.json").exists()
            else None,
            "split_counts": {
                "train_days": int(all_profile["train"].shape[0]),
                "val_days": int(all_profile["val"].shape[0]),
                "test_days": int(all_profile["test"].shape[0]),
            },
            "scaling": dataset_export_manifest.get("scaling") if dataset_export_manifest is not None else None,
            "export_fingerprint": dataset_export_manifest.get("export_fingerprint") if dataset_export_manifest is not None else None,
        }
        if config.data.scaling_factor is not None:
            dataset_patch["scaling_factor_used"] = [float(value) for value in config.data.scaling_factor]

        merge_run_manifest(
            run_layout.root,
            {
                "dataset": dataset_patch,
                "model": {
                    "model_class": config.model.model_class,
                    "dim_base": int(config.model.dim_base),
                    "num_attn_head": int(config.model.num_attn_head),
                    "num_decoder_layer": int(getattr(config.model, "num_decoder_layer", 0) or 0),
                    "dim_feedforward": int(config.model.dim_feedforward),
                    "num_parameter": int(config.model.num_parameter) if config.model.num_parameter is not None else None,
                    "num_channel": int(num_channel),
                    "seq_length": int(seq_length),
                },
            },
        )

    if config.model.resume:
        if config.model.load_time_id is None or config.model.load_milestone is None:
            logger.warning("resume=True but load_time_id or load_milestone is missing. Proceeding without resume.")
        else:
            load_milestone = config.model.load_milestone
            load_time_id = config.model.load_time_id
            if conditioning:
                load_run_id = f"train-diffusion-{diffusion_objective}-cond-{load_time_id}"
            else:
                load_run_id = f"train-diffusion-{diffusion_objective}-{train_season}-{load_time_id}"
            trainer.load_model(
                milestone=load_milestone,
                directory=trainer.result_folder,
                log_id=load_run_id,
                ignore_init_final=True,
            )
            logger.info("Loaded model milestone=%s run_id=%s", load_milestone, load_time_id)

    config.model.load_time_id = time_id
    save_config(config, time_id)
    config_save_path = artifact_paths.config_dir / f"exp_config_{time_id}.yaml"
    logger.info("Saved configuration to %s", config_save_path)
    if is_main_process:
        merge_run_manifest(
            run_layout.root,
            {
                "paths": {"config_path": str(config_save_path)},
                "status": {"training": "running"},
            },
        )

    logger.info("Training initiated.")
    trainer.train()
    logger.info("Training complete.")
    trainer.accelerator.wait_for_everyone()

    if trainer.accelerator.is_main_process:
        from energydiff.utils.sample import ancestral_sample, dpm_solver_sample

        logger.info("Post-training sampling initiated.")
        if config.sample.dpm_solver_sample and not config.diffusion.use_rectified_flow:
            dpm_sampler = DPMSolverSampler(trainer.ema.ema_model)
            generated_samples = dpm_solver_sample(
                dpm_sampler,
                total_num_sample=config.sample.num_sample,
                batch_size=config.sample.val_batch_size,
                step=config.sample.num_sampling_step,
                shape=(num_channel, seq_length),
                conditioning=None,
                cfg_scale=config.sample.cfg_scale,
                accelerator=None,
            )
        else:
            generated_samples = ancestral_sample(
                config.sample.num_sample,
                config.sample.val_batch_size,
                cond=None,
                cfg_scale=config.sample.cfg_scale,
                model=trainer.ema.ema_model,
            )

        target = torch.cat([all_profile["val"], all_profile["test"]], dim=0)
        if config.data.vectorize:
            generated_samples = dataset_collection.inverse_vectorize_fn(
                generated_samples, style=config.data.style_vectorize
            )
            target = dataset_collection.inverse_vectorize_fn(target, style=config.data.style_vectorize)

        _config = deepcopy(config)
        _config.time_id = time_id
        _config.model.load_time_id = time_id
        filename = get_generated_filename(_config, model="ddpm")
        save_path = artifact_paths.samples_dir / filename
        torch.save(generated_samples, save_path)
        logger.info("Saved synthetic samples to %s", save_path)

        source = generated_samples[:2000].cpu()
        target = target[:].cpu()
        eval_res = {}
        for metric_name, metric_fn in dict_eval_fn.items():
            eval_res[metric_name] = metric_fn(source, target)
            logger.info("%s: %.4f", metric_name, eval_res[metric_name])

        merge_run_manifest(
            run_layout.root,
            {
                "status": {"training": "completed"},
                "artifacts": {
                    "final_sample": str(save_path),
                },
            },
        )

        if log_wandb and wandb is not None:
            wandb.log({"Test": eval_res})
    else:
        raise SystemExit(0)


if __name__ == "__main__":
    main()

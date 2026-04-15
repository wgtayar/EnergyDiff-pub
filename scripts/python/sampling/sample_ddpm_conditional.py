from __future__ import annotations

from copy import deepcopy
from multiprocessing import cpu_count
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_SRC_DIR = REPO_ROOT / "src" / "data"
if str(DATA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SRC_DIR))

from experiment_layout import merge_run_manifest

from energydiff.dataset import NAME_SEASONS, PIT
from energydiff.diffusion import Trainer1D
from energydiff.diffusion.dataset import ConditionalDataset1D
from energydiff.utils.argument_parser import inference_parser
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
from energydiff.utils.sample import ConditionCrafter, ancestral_sample


def absolutize_config_paths(config) -> None:
    if config.data.root is not None:
        config.data.root = str(Path(config.data.root).expanduser().resolve())
    if config.artifact_root is not None:
        config.artifact_root = str(Path(config.artifact_root).expanduser().resolve())
    if config.run_root is not None:
        config.run_root = str(Path(config.run_root).expanduser().resolve())


def main():
    config = inference_parser()
    absolutize_config_paths(config)
    season = config.data.val_season
    if config.model.load_time_id is None:
        raise ValueError("load_time_id is not specified.")

    conditioning = config.model.conditioning
    cfg_scale = config.sample.cfg_scale
    diffusion_objective = config.diffusion.prediction_type
    num_sampling_step = config.sample.num_sampling_step
    val_batch_size = config.sample.val_batch_size
    load_time_id = config.model.load_time_id
    load_milestone = config.model.load_milestone
    season_or_cond = config.data.train_season if not config.model.conditioning else "cond"

    run_id = f"train-diffusion-{diffusion_objective}-{season_or_cond}-{load_time_id}"

    run_root = Path(config.run_root).expanduser() if config.run_root else Path(config.artifact_root).expanduser()
    artifact_paths = ensure_artifact_dirs(run_root)
    is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
    logger = configure_root_logger(
        artifact_paths.logs_dir / f"sample-{run_id}.log" if is_main_process else None
    )
    logger.info("Sampling run starts: %s", run_id)
    logger.info("Run root: %s", artifact_paths.root)
    logger.info("Dataset root: %s", Path(config.data.root).resolve())
    logger.info(
        "Sampling config: num_sample=%s num_sampling_step=%s batch_size=%s load_milestone=%s",
        config.sample.num_sample,
        config.sample.num_sampling_step,
        config.sample.val_batch_size,
        config.model.load_milestone,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_summary(
        logger,
        amp_enabled=config.train.amp,
        mixed_precision_type=config.train.mixed_precision_type,
    )

    dataset_collection = create_dataset(config.data)
    pre_transforms = {}
    pit: PIT = dataset_collection.dataset.pit
    if pit is not None:
        pre_transforms["pit"] = pit.transform
        foobar = deepcopy(config.data)
        foobar.pit = False
        dataset_collection = create_dataset(foobar)

    all_profile, all_condition = get_task_profile_condition(
        dataset_collection,
        season=season,
        conditioning=conditioning,
        area=getattr(config.data, "val_area", None),
        lcl_use_fraction=config.data.lcl_use_fraction,
    )

    testset = ConditionalDataset1D(all_profile["test"], all_condition["test"])
    logger.info("Test dataset prepared with %s samples.", int(all_profile["test"].shape[0]))

    cond_embedder = None
    if conditioning:
        cond_embedder = create_cond_embedder_wrapped(
            dataset_collection=dataset_collection,
            dim_embedding=config.model.dim_base,
        )
        list_num_emb = [embedder.num_embedding for embedder in cond_embedder.list_embedder]
        dict_cond_num_emb = {
            cond_name: cond_num_emb
            for cond_name, cond_num_emb in zip(dataset_collection.dict_cond_dim.keys(), list_num_emb)
        }
    else:
        dict_cond_num_emb = {}

    num_channel = testset.num_channel
    seq_length = testset.sequence_length
    backbone_model = create_backbone(
        config.model,
        num_in_channel=num_channel,
        cond_embedder=cond_embedder,
    ).to(device)

    if not config.diffusion.use_rectified_flow:
        diffusion = create_diffusion(
            backbone_model,
            seq_length=seq_length,
            ddpm_config=config.diffusion,
            num_sampling_timestep=num_sampling_step,
        ).to(device)
        diffusion.compile()
    else:
        diffusion = create_rectified_flow(
            base_model=backbone_model,
            seq_length=seq_length,
            rf_config=config.diffusion,
            num_discretization_step=num_sampling_step,
        ).to(device)

    trainer = Trainer1D.from_config(
        config.train,
        diffusion_model=diffusion,
        spaced_diffusion_model=diffusion,
        dataset=testset,
        val_dataset=testset,
        log_id=run_id,
        num_dataloader_workers=int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", cpu_count())),
        log_wandb=False,
        distribute_ema=True,
        result_folder=str(artifact_paths.checkpoints_dir),
    )
    if trainer.accelerator.is_main_process:
        logger.info("Sampling %s samples.", config.sample.num_sample)
    trainer.load(milestone=load_milestone)
    logger.info("Loaded checkpoint milestone=%s from %s", load_milestone, artifact_paths.checkpoints_dir)

    cond = None
    if conditioning:
        cond_crafter = ConditionCrafter(dict_cond_num_emb)
        dict_cond = {
            "season": torch.tensor(NAME_SEASONS.index(season)),
        }
        if config.data.dataset == "cossmic" and "area" in config.data.target_labels:
            dict_cond["area"] = torch.tensor(dataset_collection.condition_mapping["area"][config.data.val_area])
        cond = cond_crafter(batch_size=val_batch_size, dict_cond=dict_cond)

    sample_ddpm = ancestral_sample(
        config.sample.num_sample // trainer.accelerator.num_processes,
        val_batch_size,
        cond=cond,
        cfg_scale=cfg_scale if conditioning else 1.0,
        trainer=trainer,
    )
    if config.data.vectorize:
        sample_ddpm = dataset_collection.inverse_vectorize_fn(sample_ddpm, style=config.data.style_vectorize)

    if trainer.accelerator.is_main_process:
        filename = get_generated_filename(config, model="ddpm")
        save_path = artifact_paths.samples_dir / filename
        torch.save(sample_ddpm, save_path)
        logger.info("Saved generated samples to %s", save_path)
        merge_run_manifest(
            artifact_paths.root,
            {
                "artifacts": {"final_sample": str(save_path)},
            },
        )


if __name__ == "__main__":
    main()

import os
import json

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary

from config import cfg
from datasets.dexycb import DexYCBDataModule
from datasets.mvhand import MVHandDataModule
from datasets.ho3d import HO3DDataModule

from models.handmvnet import HandMvNet as Model


# Set CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = cfg["gpu_ids"]

# Setting the seed
L.seed_everything(42, workers=True)

# Set the number of GPUs
NUM_GPUS = cfg["train"]["gpus"]
VISIBLE_NUM_GPUS = torch.cuda.device_count()
assert VISIBLE_NUM_GPUS == NUM_GPUS, f"Number of GPUs mismatch! Expected: {NUM_GPUS}, Got: {VISIBLE_NUM_GPUS}"

ACCUMULATE_BATCH = 2 if ((cfg["data"]["batch_size"] < 64) and (NUM_GPUS < 3)) else 1
cfg["train"]["accumulate_batch"] = cfg["train"].get("accumulate_batch", ACCUMULATE_BATCH)


if __name__ == "__main__":
    print("Visible GPUs:", cfg["train"]["gpus"])

    checkpoint_cb = ModelCheckpoint(filename="{epoch}-{step}-{val_mpjpe:.3f}", monitor="val_mpjpe", mode="min", save_last=True)
    summary_cb = ModelSummary(max_depth=2)
    lr_monitor_cb = LearningRateMonitor("epoch")

    grad_clip = cfg["train"].get("grad_clip", 1)
    strategy = "ddp" # "ddp_find_unused_parameters_true"
    strategy = "auto" if NUM_GPUS < 2 else strategy
    trainer_args = dict(
        # fast_dev_run=2,
        accumulate_grad_batches=ACCUMULATE_BATCH,
        devices=cfg["train"]["gpus"], accelerator="gpu",
        gradient_clip_val=grad_clip,
        strategy=strategy,
        max_epochs=cfg["train"]["epochs"],
        # profiler="simple",
        # deterministic=True,
        # benchmark=False,
        callbacks=[checkpoint_cb, summary_cb, lr_monitor_cb],
        default_root_dir=cfg["base_output_dir"],
        detect_anomaly=False,
        deterministic="warn"
    )

    # initialize trainer
    trainer = L.Trainer(**trainer_args)

    rank = trainer.global_rank
    cfg["train"]["device"] = trainer.strategy.root_device
    if trainer.log_dir is not None:
        cfg["train"]["vis_dir"] = os.path.join(trainer.log_dir, "vis")
    else:
        cfg["train"]["vis_dir"] = os.path.join(cfg["base_output_dir"], "vis")

    if rank == 0:
        print("Config:", json.dumps(cfg, indent=2, default=str))
        print("\nTrainer args:", trainer_args)
        # create directories for visualizations
        os.makedirs(os.path.join(cfg["train"]["vis_dir"], "train"), exist_ok=True)
        os.makedirs(os.path.join(cfg["train"]["vis_dir"], "val"), exist_ok=True)
        os.makedirs(os.path.join(cfg["train"]["vis_dir"], "test"), exist_ok=True)

    print(f"[{rank}]: Initializing model: {cfg['name']}")
    model = Model(cfg["train"], cfg["model"], cfg["data"])

    print(f"[{rank}]: Loading data module...")
    dataset_name = cfg["data"].get("name", "dexycb")
    if dataset_name == "dexycb":
        dm = DexYCBDataModule(cfg["data"])
    elif dataset_name == "mvhand":
        dm = MVHandDataModule(cfg["data"])
    elif dataset_name == "ho3d":
        dm = HO3DDataModule(cfg["data"])
    else:
        print(f"{dataset_name} dataset not found.")
        exit()

    print(f"[{rank}]: Starting model training...")
    history = trainer.fit(model=model, datamodule=dm)

    best_model_checkpoint = checkpoint_cb.best_model_path
    if not best_model_checkpoint:
        print("[Warning] No best model checkpoint found! Using the last model...")
        val = trainer.validate(ckpt_path="last", datamodule=dm, verbose=True)
    else:
        print("Validating best model checkpoint...")
        val = trainer.validate(ckpt_path="best", datamodule=dm, verbose=True)

    # save validation results in a file
    val_file_path = os.path.join(cfg["train"]["vis_dir"],"..", "val.json")
    with open(val_file_path, "w", encoding="utf-8") as file:
        json.dump(val, file, indent=2)
    print("Validation result:", val)
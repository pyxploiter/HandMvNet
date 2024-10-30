import os
import json
import torch
import lightning as L
from collections import OrderedDict

from config import cfg
from datasets.dexycb import DexYCBDataModule
from datasets.mvhand import MVHandDataModule
from datasets.ho3d import HO3DDataModule

from models.handmvnet import HandMvNet as Model


def is_legacy_version(state_dict):
    """
    Check if the model is a legacy version based on mismatch between keys.
    """
    # Define common mismatching keys for legacy models
    legacy_keys = ["pose_net.conv.0.weight", "sample_net.conv.0.weight"]
    for key in legacy_keys:
        if key in state_dict:
            return True
    return False


def load_checkpoint_with_legacy_fix(checkpoint_path, model, device="cpu"):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract the state_dict from the checkpoint
    state_dict = checkpoint['state_dict']

    # Check if it's a legacy version by inspecting mismatched keys
    if is_legacy_version(state_dict):
        print("[warning] Legacy version detected. Remapping keys...")
        # Create a new state_dict with remapped keys
        new_state_dict = OrderedDict()
        for old_key, value in state_dict.items():
            # Replace old keys with new ones
            new_key = old_key.replace('pose_net.conv.', 'pose_net.') \
                             .replace('sample_net.', 'sample_nets.0.')
            # Add the remapped key-value pair
            new_state_dict[new_key] = value
        # Load the remapped state_dict into the model
        model.load_state_dict(new_state_dict, strict=True)
        print("[info] legacy model loaded successfully.")
    else:
        # Load state_dict as is if no legacy issues
        model.load_state_dict(state_dict, strict=True)

    return model


if __name__ == "__main__":
    cfg["data"]["batch_size"] = 16
    cfg["data"]["num_workers"] = 6
    cfg["model"]["get_vertices"] = True

    # Setting the seed
    L.seed_everything(42, workers=True)

    torch.autograd.set_detect_anomaly(True)

    cfg["train"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print("Config:", json.dumps(cfg, indent=2))

    CHECKPOINT_PATH = cfg["checkpoint"]
    if CHECKPOINT_PATH:
        # print("\nStarting evaluation...")
        trainer = L.Trainer(devices=1, accelerator="auto", logger=False)
        if trainer.log_dir is not None:
            cfg["train"]["vis_dir"] = os.path.join(trainer.log_dir, "vis")
        else:
            cfg["train"]["vis_dir"] = os.path.join(cfg["base_output_dir"], "vis")

        model = Model(
            train_params=cfg["train"],
            model_params=cfg["model"],
            data_params=cfg["data"]
        )
        print("\nLoading weights from checkpoint:", CHECKPOINT_PATH)
        # Check and fix mismatches if it's a legacy version
        model = load_checkpoint_with_legacy_fix(CHECKPOINT_PATH, model, cfg["train"]["device"])
        model.eval()
        model.freeze()

        dataset_name = cfg["data"].get("name", "dexycb")
        if dataset_name == "dexycb":
            dm = DexYCBDataModule(cfg["data"])
        elif dataset_name == "mvhand":
            dm = MVHandDataModule(cfg["data"])
        elif dataset_name == "ho3d":
            dm = HO3DDataModule(cfg["data"])
            cfg["data"]["num_workers"] = 2
            print(f"[warn] setting num_workers={cfg['data']['num_workers']}")
        else:
            print(f"{dataset_name} dataset not found.")
            exit()

        val = trainer.validate(model, datamodule=dm, verbose=True)
        json.dump(val, open(os.path.join("/".join(cfg["checkpoint"].split("/")[:-2]), "val.json"), "w"), indent=2)
        print("val:", val)

        test = trainer.test(model, datamodule=dm, verbose=True)
        json.dump(test, open(os.path.join("/".join(cfg["checkpoint"].split("/")[:-2]), "test.json"), "w"), indent=2)
        print("test:", test)
    else:
        print("Checkpoint not found at:", CHECKPOINT_PATH)
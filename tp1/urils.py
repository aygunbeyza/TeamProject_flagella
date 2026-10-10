import os
import json
from datetime import datetime

BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
REPO_DIR = "/data/horse/ws/beay097h-teamproject/TeamProject_flagella"
CSV_PATH = os.path.join(BASE_DIR, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
OUTPUTS_DIR = os.path.join(REPO_DIR, "outputs")


def make_run_dir(exp_name: str) -> str:
    """outputs/<YYYYmmdd_HHMM>_<exp_name>/{figures,logs,checkpoints}"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(OUTPUTS_DIR, f"{stamp}_{exp_name}")
    for sub in ("figures", "logs", "checkpoints"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def fig_path(run_dir: str, exp: str, step: int, what: str, **params) -> str:
    """ör: exp01_negatives_step04_sanity_check_sigma6.png"""
    tag = "_".join(f"{k}{v}" for k, v in params.items())
    name = f"{exp}_step{step:02d}_{what}" + (f"_{tag}" if tag else "") + ".png"
    return os.path.join(run_dir, "figures", name)


def save_config(run_dir: str, cfg: dict) -> None:
    with open(os.path.join(run_dir, "logs", "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

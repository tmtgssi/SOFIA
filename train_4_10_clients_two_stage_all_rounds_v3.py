import os
import glob
import time
import platform
import subprocess
import csv
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLOWorld


# ============================================================
# CONFIGURATION
# ============================================================

# Hyperparameters:
# /home/gssi-lab/Documents/trieu/4_10_clients/
#   Ours_wi_SO_wi_inpainting_NEU/FL_RESULTS/
#       4_CLIENTS/round_1/client_1/best_params.yaml
#       ...
#       10_CLIENTS/round_5/client_10/best_params.yaml
HYPERPARAM_ROOT = Path(
    "/home/gssi-lab/Documents/trieu/4_10_clients/"
    "Ours_wi_SO_wi_inpainting_NEU/FL_RESULTS"
)

# Same pretrained model used by the first training stage in c1c2c3.ipynb.
BASE_MODEL_FILE = Path("./yolov8x-worldv2.pt")

# Stage 1: inpainting-reconstructed dataset, e.g.
#   ./client_1_inpainting_reconstructed/data.yaml
# Stage 2: regular client dataset, e.g.
#   ./client_1/data.yaml

# All generated checkpoints/results go here so different
# client-count / round combinations never overwrite each other.
OUTPUT_ROOT = Path("./ckpt_4_10_clients_two_stage")

IMGSZ = 224
VERBOSE = True

CLIENT_COUNTS = range(4, 11)  # 4 ... 10
ROUNDS = range(1, 6)          # round_1 ... round_5

# Use SGD explicitly so lr0 and momentum from best_params.yaml
# are actually used. The original notebook used optimizer="auto",
# which reported that it ignored supplied lr0/momentum.


# ============================================================
# PATH HELPERS
# ============================================================

def hp_path(client_count, round_id, client_id):
    return (
        HYPERPARAM_ROOT
        / f"{client_count}_CLIENTS"
        / f"round_{round_id}"
        / f"client_{client_id}"
        / "best_params.yaml"
    )


def stage1_data_yaml(client_id):
    # Stage 1 uses the inpainting-reconstructed dataset.
    # Example: ./client_1_inpainting_reconstructed/data.yaml
    return Path(f"./client_{client_id}_inpainting_reconstructed/data.yaml")


def stage2_data_yaml(client_id):
    # Stage 2 uses the regular client dataset.
    # Example: ./client_1/data.yaml
    return Path(f"./client_{client_id}/data.yaml")


def run_root(client_count, round_id, client_id):
    return (
        OUTPUT_ROOT
        / f"{client_count}_CLIENTS"
        / f"round_{round_id}"
        / f"client_{client_id}"
    )


def stage1_checkpoint(client_count, round_id, client_id):
    return (
        run_root(client_count, round_id, client_id)
        / "stage1"
        / f"client_{client_id}_stage1.pt"
    )


def stage2_checkpoint(client_count, round_id, client_id):
    return (
        run_root(client_count, round_id, client_id)
        / "stage2"
        / f"client_{client_id}_local_model.pt"
    )


# ============================================================
# LOAD HYPERPARAMETERS
# ============================================================

def load_best_params(client_count, round_id, client_id):
    path = hp_path(client_count, round_id, client_id)

    if not path.is_file():
        raise FileNotFoundError(
            f"Hyperparameter file not found:\n{path}"
        )

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML: {path}")

    # Check the metadata if present.
    expected = {
        "client_count": client_count,
        "client_id": client_id,
        "round": round_id,
    }

    for key, expected_value in expected.items():
        if key in cfg and int(cfg[key]) != expected_value:
            raise ValueError(
                f"{path}: {key}={cfg[key]} "
                f"but expected {expected_value}"
            )

    hp = cfg.get("hyperparameters")
    if not isinstance(hp, dict):
        raise ValueError(
            f"{path}: missing 'hyperparameters' section"
        )

    required = [
        "epochs",
        "batch",
        "lr0",
        "momentum",
        "weight_decay",
        "warmup_epochs",
    ]

    missing = [k for k in required if k not in hp]
    if missing:
        raise ValueError(
            f"{path}: missing hyperparameters {missing}"
        )

    return cfg


# ============================================================
# DATA HELPERS
# ============================================================

def resolve_data_yaml(data_yaml):
    data_yaml = Path(data_yaml)

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"DATA_YAML not found:\n{data_yaml.resolve()}"
        )

    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset YAML: {data_yaml}")

    return data


def resolve_split_dir(data_yaml, split_value):
    split_path = Path(split_value)

    if split_path.is_absolute():
        return split_path

    # This handles data.yaml entries such as ./train and ./val.
    return data_yaml.parent / split_path


def count_images(path):
    if not Path(path).is_dir():
        return 0

    extensions = ("*.jpg", "*.jpeg", "*.png")
    return sum(
        len(glob.glob(str(Path(path) / ext)))
        for ext in extensions
    )


def count_instances(path):
    if not Path(path).is_dir():
        return 0

    total = 0
    for txt in glob.glob(str(Path(path) / "*.txt")):
        with open(txt, "r") as f:
            total += sum(1 for line in f if line.strip())
    return total


def dataset_stats(data_yaml):
    data = resolve_data_yaml(data_yaml)

    train_dir = resolve_split_dir(data_yaml, data["train"])
    val_dir = resolve_split_dir(data_yaml, data["val"])

    stats = {
        "names": data["names"],
        "nc": data.get("nc", len(data["names"])),
        "train_images": count_images(train_dir / "images"),
        "train_instances": count_instances(train_dir / "labels"),
        "val_images": count_images(val_dir / "images"),
        "val_instances": count_instances(val_dir / "labels"),
    }

    return data, stats


# ============================================================
# MODEL HELPER
# ============================================================

def prepare_model(model_file, num_classes):
    model = YOLOWorld(str(model_file))

    model.model.nc = num_classes
    model.model.model[-1].nc = num_classes
    model.model.model[-1].no = num_classes + 5

    return model


# ============================================================
# ONE TRAINING STAGE
# ============================================================

def train_stage(
    *,
    client_count,
    round_id,
    client_id,
    stage_name,
    model_file,
    data_yaml,
    hp,
    output_dir,
    epochs_override=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    data, stats = dataset_stats(data_yaml)
    num_classes = len(data["names"])

    epochs = (
        int(epochs_override)
        if epochs_override is not None
        else int(hp["epochs"])
    )

    batch = int(hp["batch"])
    lr0 = float(hp["lr0"])
    momentum = float(hp["momentum"])
    weight_decay = float(hp["weight_decay"])
    warmup_epochs = int(hp["warmup_epochs"])

    print("\n" + "=" * 90)
    print(
        f"{stage_name.upper()} | "
        f"{client_count} CLIENTS | "
        f"ROUND {round_id} | "
        f"CLIENT {client_id}"
    )
    print("=" * 90)

    print(f"Model:      {model_file}")
    print(f"DATA_YAML:  {data_yaml}")
    print(f"Classes:    {num_classes}")
    print(f"Data stats: {stats}")
    print(f"Hyperparams: {hp}")

    model = prepare_model(model_file, num_classes)

    start_time = time.time()

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=batch,
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        verbose=VERBOSE

    )

    training_time = time.time() - start_time

    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "training_time_sec": round(training_time, 2),
        "val_images": stats["val_images"],
        "val_instances": stats["val_instances"],
    }

    print("\n===== Training Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return model, results, metrics, stats


# ============================================================
# TRAIN ONE CLIENT FOR ONE ROUND
# ============================================================

def train_client_round(client_count, round_id, client_id):
    root = run_root(client_count, round_id, client_id)
    root.mkdir(parents=True, exist_ok=True)

    hp_file = hp_path(client_count, round_id, client_id)
    cfg = load_best_params(client_count, round_id, client_id)
    hp = cfg["hyperparameters"]

    data1 = stage1_data_yaml(client_id)
    data2 = stage2_data_yaml(client_id)

    ckpt1 = stage1_checkpoint(client_count, round_id, client_id)
    ckpt2 = stage2_checkpoint(client_count, round_id, client_id)

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------
    #
    # Original pattern:
    #   LOCAL_MODEL_FILE = "./yolov8x-worldv2.pt"
    #   DATA_YAML = "./client_1/data.yaml"
    #
    # We preserve that structure, but create a unique checkpoint
    # for every client-count / round / client combination.
    #
    if ckpt1.is_file():
        print(f"[SKIP STAGE 1] {ckpt1}")
    else:
        stage1_dir = root / "stage1"

        model1, results1, metrics1, stats1 = train_stage(
            client_count=client_count,
            round_id=round_id,
            client_id=client_id,
            stage_name="stage 1",
            model_file=BASE_MODEL_FILE,
            data_yaml=data1,
            hp=hp,
            output_dir=stage1_dir,
            epochs_override=60,
        )
        model1.save(str(ckpt1))
        print(f"\nStage-1 checkpoint saved to:\n{ckpt1}")

        with open(stage1_dir / "metrics.yaml", "w") as f:
            yaml.safe_dump(metrics1, f, sort_keys=False)

    # The second stage MUST start from the checkpoint produced
    # by stage 1, not from yolov8x-worldv2.pt.
    if not ckpt1.is_file():
        raise FileNotFoundError(
            f"Stage-1 checkpoint required for stage 2 was not found:\n{ckpt1}"
        )

    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------
    #
    # Original pattern:
    #   LOCAL_MODEL_FILE = "./ckpt_local/client_1_local_model.pt"
    #   DATA_YAML = "/home/.../FL_boost/.../client_1/data.yaml"
    #
    # Here LOCAL_MODEL_FILE is the unique stage-1 checkpoint.
    #
    if ckpt2.is_file():
        print(f"[SKIP STAGE 2] {ckpt2}")
        metrics2 = {}
        stats2 = {}
    else:
        stage2_dir = root / "stage2"

        model2, results2, metrics2, stats2 = train_stage(
            client_count=client_count,
            round_id=round_id,
            client_id=client_id,
            stage_name="stage 2",
            model_file=ckpt1,
            data_yaml=data2,
            hp=hp,
            output_dir=stage2_dir,
            epochs_override=100,
        )
        model2.save(str(ckpt2))
        print(f"\nStage-2/final checkpoint saved to:\n{ckpt2}")
        
        # Remove Stage-1 training files after Stage 2 is successfully saved.
        stage1_dir = root / "stage1"
        stage2train = root / "stage2" /"train"
        if stage1_dir.exists():
        	shutil.rmtree(stage1_dir)
        	shutil.rmtree(stage2train)
        print(f"Removed all Stage-1 + Stage 2 train folder")

        with open(stage2_dir / "metrics.yaml", "w") as f:
            yaml.safe_dump(metrics2, f, sort_keys=False)

    # Save the complete provenance/configuration.
    run_config = {
        "client_count": client_count,
        "client_id": client_id,
        "round": round_id,
        "hyperparameter_file": str(hp_file),
        "best_score": cfg.get("best_score"),
        "hyperparameters": hp,
        "imgsz": IMGSZ,
        "stage1": {
            "model_input": str(BASE_MODEL_FILE),
            "data_yaml": str(data1),
            "checkpoint_output": str(ckpt1),
            "metrics": metrics1 if "metrics1" in locals() else {},
        },
        "stage2": {
            "model_input": str(ckpt1),
            "data_yaml": str(data2),
            "checkpoint_output": str(ckpt2),
            "metrics": metrics2,
        },
    }

    with open(root / "run_config.yaml", "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    print("\n" + "-" * 90)
    print(
        f"FINISHED: {client_count} clients | "
        f"round {round_id} | client {client_id}"
    )
    print(f"Stage 1: {ckpt1}")
    print(f"Stage 2: {ckpt2}")
    print("-" * 90)

    return {
        "client_count": client_count,
        "round": round_id,
        "client_id": client_id,
        "status": "completed",
        "best_score": cfg.get("best_score"),
        "stage1_precision": metrics1.get("precision", ""),
        "stage1_recall": metrics1.get("recall", ""),
        "stage1_map50": metrics1.get("map50", ""),
        "stage1_map50_95": metrics1.get("map50_95", ""),
        "stage1_time_sec": metrics1.get("training_time_sec", ""),
        "stage2_precision": metrics2.get("precision", ""),
        "stage2_recall": metrics2.get("recall", ""),
        "stage2_map50": metrics2.get("map50", ""),
        "stage2_map50_95": metrics2.get("map50_95", ""),
        "stage2_time_sec": metrics2.get("training_time_sec", ""),
        "stage1_checkpoint": str(ckpt1),
        "stage2_checkpoint": str(ckpt2),
    }


# ============================================================
# RUN ALL 7 CASES x 5 ROUNDS x ALL CLIENTS
# ============================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

total_runs = sum(CLIENT_COUNTS) * len(list(ROUNDS))
run_number = 0
all_results = []

print("=" * 90)
print("TWO-STAGE TRAINING")
print("=" * 90)
print(f"Client cases: 4, 5, 6, 7, 8, 9, 10")
print(f"Rounds:       1, 2, 3, 4, 5")
print(f"Runs:         {total_runs}")
print(f"Trainings:    {total_runs * 2} stages")
print()
print("For EACH client/round:")
print("  Stage 1: yolov8x-worldv2.pt + ./client_N/data.yaml")
print("  Stage 2: stage-1 checkpoint + absolute FL_boost/client_N/data.yaml")
print("=" * 90)

for client_count in CLIENT_COUNTS:
    for round_id in ROUNDS:
        for client_id in range(1, client_count + 1):
            run_number += 1

            print(
                f"\n\n### RUN {run_number}/{total_runs}: "
                f"{client_count} CLIENTS | "
                f"ROUND {round_id} | "
                f"CLIENT {client_id}"
            )

            try:
                result = train_client_round(
                    client_count=client_count,
                    round_id=round_id,
                    client_id=client_id,
                )
                all_results.append(result)

            except Exception as exc:
                print(
                    f"\n[ERROR] {client_count} clients | "
                    f"round {round_id} | "
                    f"client {client_id}\n{exc}"
                )

                all_results.append({
                    "client_count": client_count,
                    "round": round_id,
                    "client_id": client_id,
                    "status": "failed",
                    "error": repr(exc),
                })

# ============================================================
# SUMMARY CSV
# ============================================================

summary_csv = OUTPUT_ROOT / "training_summary.csv"

columns = [
    "client_count",
    "round",
    "client_id",
    "status",
    "best_score",
    "stage1_precision",
    "stage1_recall",
    "stage1_map50",
    "stage1_map50_95",
    "stage1_time_sec",
    "stage2_precision",
    "stage2_recall",
    "stage2_map50",
    "stage2_map50_95",
    "stage2_time_sec",
    "stage1_checkpoint",
    "stage2_checkpoint",
    "error",
]

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=columns,
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(all_results)

completed = sum(
    r.get("status") == "completed" for r in all_results
)
failed = sum(
    r.get("status") == "failed" for r in all_results
)

print("\n" + "=" * 90)
print("ALL TWO-STAGE TRAINING FINISHED")
print("=" * 90)
print(f"Completed client/round runs: {completed}")
print(f"Failed client/round runs:    {failed}")
print(f"Training stages attempted:   {len(all_results) * 2}")
print(f"Summary CSV:                  {summary_csv}")
print(f"Output root:                  {OUTPUT_ROOT}")


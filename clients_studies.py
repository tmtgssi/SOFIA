# ============================================================
# FEDERATED LEARNING SIMULATION - ALL IN ONE CELL
# 30 -> 27 -> 24 -> ... -> 3 CLIENTS
# Sequential Client Training
# Hyperparameter Optimization Per Client
#pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# ============================================================

import os
import glob
import time
import random
import hashlib
import platform
import json
import csv
import copy
import traceback

import torch
import yaml
from ultralytics import YOLOWorld


# ============================================================
# 1. GLOBAL CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Base YOLO-World model
# ------------------------------------------------------------
LOCAL_MODEL_FILE = "./yolov8x-worldv2.pt"

# ------------------------------------------------------------
# Client datasets
# Expected:
#
# client_1/
#     data.yaml
#     train/
#     val/
#
# client_2/
#     data.yaml
#     train/
#     val/
#
# ...
#
# client_30/
#     data.yaml
# ------------------------------------------------------------

CLIENT_ROOT = "."

# ------------------------------------------------------------
# Federated learning configuration
# ------------------------------------------------------------

CLIENT_COUNTS = [10]
# => [30, 27, 24, 21, 18, 15, 12, 9, 6, 3]

ROUNDS = 5

IMGSZ = 224
VERBOSE = False

# ------------------------------------------------------------
# Hyperparameter optimization
# ------------------------------------------------------------

POP_SIZE = 5
MAX_ITER = 3

HYPERPARAM_SPACE = {
    "epochs":        [1, 5, 10],
    "batch":         [2,3,4],
    "lr0":           [0.001, 0.005, 0.01],
    "momentum":      [0.900, 0.937, 0.950],
    "weight_decay":  [0.0001, 0.0005, 0.001],
    "warmup_epochs": [2, 3, 4],
}

# ------------------------------------------------------------
# Random seed
# ------------------------------------------------------------

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=" * 80)
print("FEDERATED LEARNING SIMULATION")
print("=" * 80)
print("Client counts:", CLIENT_COUNTS)
print("Rounds:", ROUNDS)
print("Population:", POP_SIZE)
print("Max optimizer iterations:", MAX_ITER)
print("Image size:", IMGSZ)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 80)


# ============================================================
# 2. OUTPUT DIRECTORIES
# ============================================================

RESULTS_ROOT = "./FL_RESULTS"

os.makedirs(RESULTS_ROOT, exist_ok=True)

print(f"[INFO] Results directory: {RESULTS_ROOT}")


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def file_sha256(path):
    """
    Calculate SHA256 checksum of a file.
    """
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def count_images(path):
    """
    Count jpg/png/jpeg images recursively.
    """
    if not os.path.exists(path):
        return 0

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

    total = 0

    for ext in extensions:
        total += len(glob.glob(os.path.join(path, "**", ext), recursive=True))

    return total


def count_instances(path):
    """
    Count YOLO label instances.
    """
    if not os.path.exists(path):
        return 0

    total = 0

    for txt in glob.glob(
        os.path.join(path, "**", "*.txt"),
        recursive=True
    ):
        try:
            with open(txt, "r", encoding="utf-8") as f:
                total += sum(
                    1 for line in f
                    if line.strip()
                )
        except Exception:
            pass

    return total


def get_cpu_info():
    try:
        if platform.system() == "Windows":
            import subprocess

            output = subprocess.check_output(
                "wmic cpu get Name",
                shell=True
            ).decode(errors="ignore")

            lines = output.split("\n")

            if len(lines) > 1:
                return lines[1].strip()

        return platform.processor()

    except Exception:
        return platform.processor()


# ============================================================
# 4. CLIENT DATASET INFORMATION
# ============================================================

def get_client_yaml(client_id):
    return os.path.join(
        CLIENT_ROOT,
        f"client_{client_id}",
        "data.yaml"
    )


def load_client_data(client_id):
    yaml_path = get_client_yaml(client_id)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Missing data.yaml for client {client_id}: {yaml_path}"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


def print_client_dataset_info(client_id):

    client_dir = os.path.join(
        CLIENT_ROOT,
        f"client_{client_id}"
    )

    yaml_path = os.path.join(
        client_dir,
        "data.yaml"
    )

    if not os.path.exists(yaml_path):
        print(
            f"[WARNING] Client {client_id}: "
            f"data.yaml not found"
        )
        return

    data = load_client_data(client_id)

    train_path = os.path.join(
        client_dir,
        str(data.get("train", "train")),
        "images"
    )

    val_path = os.path.join(
        client_dir,
        str(data.get("val", "val")),
        "images"
    )

    train_label_path = os.path.join(
        client_dir,
        str(data.get("train", "train")),
        "labels"
    )

    val_label_path = os.path.join(
        client_dir,
        str(data.get("val", "val")),
        "labels"
    )

    train_images = count_images(train_path)
    val_images = count_images(val_path)

    train_instances = count_instances(
        train_label_path
    )

    val_instances = count_instances(
        val_label_path
    )

    print(
        f"Client {client_id:02d} | "
        f"Train images={train_images} | "
        f"Train instances={train_instances} | "
        f"Val images={val_images} | "
        f"Val instances={val_instances}"
    )


# ============================================================
# 5. CHECK CLIENTS
# ============================================================

print("\nChecking client datasets...")

missing_clients = []

for client_id in range(1, 31):

    yaml_path = get_client_yaml(client_id)

    if not os.path.exists(yaml_path):
        missing_clients.append(client_id)

if missing_clients:

    print("\n[WARNING]")
    print(
        "The following client data.yaml files are missing:"
    )
    print(missing_clients)

    print(
        "\nThe program will only work for client IDs "
        "that actually exist."
    )

else:

    print("[OK] client_1 ... client_30 found.")


# ============================================================
# 6. MODEL CREATION
# ============================================================

def create_model():

    model = YOLOWorld(LOCAL_MODEL_FILE)

    return model


# ============================================================
# 7. LOAD WEIGHTS INTO YOLO MODEL
# ============================================================

def load_weights_into_model(model, state_dict):

    # Copy state dict so we do not accidentally modify it
    clean_state = {}

    for key, value in state_dict.items():

        if torch.is_tensor(value):

            clean_state[key] = value

    result = model.model.load_state_dict(
        clean_state,
        strict=False
    )

    return result


# ============================================================
# 8. EXTRACT STATE DICT
# ============================================================

def extract_state_dict(model):

    state = {}

    for key, value in model.model.state_dict().items():

        if torch.is_tensor(value):

            state[key] = value.detach().cpu().clone()

    return state


# ============================================================
# 9. SAVE STATE DICT
# ============================================================

def save_state_dict(state_dict, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    torch.save(
        {
            "model": state_dict
        },
        path
    )


# ============================================================
# 10. LOAD STATE DICT FROM FILE
# ============================================================

def load_state_dict_file(path):

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(ckpt, dict) and "model" in ckpt:

        state = ckpt["model"]

        if hasattr(state, "state_dict"):
            state = state.state_dict()

        return {
            k: v.cpu().clone()
            for k, v in state.items()
            if torch.is_tensor(v)
        }

    if hasattr(ckpt, "state_dict"):

        state = ckpt.state_dict()

        return {
            k: v.cpu().clone()
            for k, v in state.items()
            if torch.is_tensor(v)
        }

    return {
        k: v.cpu().clone()
        for k, v in ckpt.items()
        if torch.is_tensor(v)
    }


# ============================================================
# 11. SAVE BEST PARAMETERS
# ============================================================

def save_best_params(
    client_count,
    client_id,
    round_idx,
    best_cfg,
    best_score
):

    directory = os.path.join(
        RESULTS_ROOT,
        f"{client_count}_CLIENTS",
        f"round_{round_idx}",
        f"client_{client_id}"
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    yaml_path = os.path.join(
        directory,
        "best_params.yaml"
    )

    output = {
        "client_count": client_count,
        "client_id": client_id,
        "round": round_idx,
        "best_score": float(best_score),
        "hyperparameters": best_cfg
    }

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            output,
            f,
            sort_keys=False
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_path = os.path.join(
        directory,
        "best_params.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    return yaml_path, json_path


# ============================================================
# 12. SAVE LOCAL MODEL
# ============================================================

def save_local_model(
    model,
    client_count,
    client_id,
    round_idx
):

    directory = os.path.join(
        RESULTS_ROOT,
        f"{client_count}_CLIENTS",
        f"round_{round_idx}",
        "local_models"
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    path = os.path.join(
        directory,
        f"client_{client_id}_local.pt"
    )

    state_dict = extract_state_dict(model)

    save_state_dict(
        state_dict,
        path
    )

    return path


# ============================================================
# 13. SAVE GLOBAL MODEL
# ============================================================

def save_global_model(
    state_dict,
    client_count,
    round_idx
):

    directory = os.path.join(
        RESULTS_ROOT,
        f"{client_count}_CLIENTS",
        "global_models"
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    path = os.path.join(
        directory,
        f"global_model_round_{round_idx}.pt"
    )

    save_state_dict(
        state_dict,
        path
    )

    return path


# ============================================================
# 14. HYPERPARAMETER NEIGHBOR
# ============================================================

def random_neighbor(cfg):

    new_cfg = cfg.copy()

    key = random.choice(
        list(HYPERPARAM_SPACE.keys())
    )

    new_cfg[key] = random.choice(
        HYPERPARAM_SPACE[key]
    )

    return new_cfg


# ============================================================
# 15. CONFIG KEY
# ============================================================

def config_key(cfg):

    return tuple(
        (k, cfg[k])
        for k in sorted(cfg.keys())
    )


# ============================================================
# 16. EVALUATE HYPERPARAMETER CONFIG
# ============================================================

def evaluate_config(
    cfg,
    client_id,
    global_state
):

    try:

        data_yaml = get_client_yaml(
            client_id
        )

        data = load_client_data(
            client_id
        )

        model = create_model()

        # ----------------------------------------------------
        # Load global model
        # ----------------------------------------------------

        load_weights_into_model(
            model,
            global_state
        )

        # ----------------------------------------------------
        # Number of classes
        # ----------------------------------------------------

        num_classes = len(
            data["names"]
        )

        try:

            model.model.nc = num_classes

            model.model.model[-1].nc = num_classes

            model.model.model[-1].no = (
                num_classes + 5
            )

        except Exception:

            pass

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        results = model.train(

            data=data_yaml,

            epochs=cfg["epochs"],

            imgsz=IMGSZ,

            batch=cfg["batch"],

            lr0=cfg["lr0"],

            momentum=cfg["momentum"],

            weight_decay=cfg["weight_decay"],

            warmup_epochs=cfg["warmup_epochs"],

            verbose=False,

            mosaic=0
        )

        # ----------------------------------------------------
        # Fitness
        # ----------------------------------------------------

        try:

            precision = float(
                results.box.mp
            )

        except Exception:

            precision = 0.0

        try:

            recall = float(
                results.box.mr
            )

        except Exception:

            recall = 0.0

        try:

            map50 = float(
                results.box.map50
            )

        except Exception:

            map50 = 0.0

        try:

            map5095 = float(
                results.box.map
            )

        except Exception:

            map5095 = 0.0

        score = (
            precision
            + recall
            + map50
            + map5095
        )

        return score

    except Exception as e:

        print(
            f"[Client {client_id}] "
            f"Evaluation error: {e}"
        )

        return 0.0


# ============================================================
# 17. SWARM HORSE OPTIMIZATION
# ============================================================

def swarm_horse_optimization(
    client_id,
    global_state,
    client_count,
    round_idx
):

    print(
        f"\n[SHO] Client {client_id} | "
        f"{client_count} clients | "
        f"Round {round_idx}"
    )

    population = []

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    for _ in range(POP_SIZE):

        cfg = {
            k: random.choice(v)
            for k, v in HYPERPARAM_SPACE.items()
        }

        population.append(cfg)

    fitness = {}

    # --------------------------------------------------------
    # Evaluate initial population
    # --------------------------------------------------------

    for idx, cfg in enumerate(population):

        key = config_key(cfg)

        if key not in fitness:

            score = evaluate_config(
                cfg,
                client_id,
                global_state
            )

            fitness[key] = score

        print(
            f"[SHO] Client {client_id} "
            f"Initial {idx + 1}/{POP_SIZE} "
            f"score={fitness[key]:.6f}"
        )

    # --------------------------------------------------------
    # Best
    # --------------------------------------------------------

    best_cfg = max(
        population,
        key=lambda c: fitness[config_key(c)]
    )

    best_score = fitness[
        config_key(best_cfg)
    ]

    print(
        f"[SHO] Client {client_id} "
        f"Initial BEST={best_score:.6f}"
    )

    # --------------------------------------------------------
    # Optimization iterations
    # --------------------------------------------------------

    for iteration in range(
        1,
        MAX_ITER + 1
    ):

        new_population = []

        for cfg in population:

            proposal = random_neighbor(
                cfg
            )

            proposal_key = config_key(
                proposal
            )

            current_key = config_key(
                cfg
            )

            if proposal_key not in fitness:

                fitness[
                    proposal_key
                ] = evaluate_config(
                    proposal,
                    client_id,
                    global_state
                )

            if (
                fitness[proposal_key]
                >=
                fitness[current_key]
            ):

                new_population.append(
                    proposal
                )

            else:

                new_population.append(
                    cfg
                )

        population = new_population

        current_best = max(
            population,
            key=lambda c: fitness[
                config_key(c)
            ]
        )

        current_score = fitness[
            config_key(current_best)
        ]

        if current_score > best_score:

            best_cfg = current_best.copy()

            best_score = current_score

        print(
            f"[SHO] Client {client_id} "
            f"Iteration "
            f"{iteration}/{MAX_ITER} | "
            f"BEST={best_score:.6f}"
        )

    print(
        f"[SHO] Client {client_id} FINAL BEST:"
    )

    print(best_cfg)

    print(
        f"[SHO] Score={best_score:.6f}"
    )

    return best_cfg, best_score


# ============================================================
# 18. FEDERATED WEIGHT AGGREGATION
# ============================================================

def average_weights(
    client_state_dicts
):

    if not client_state_dicts:

        raise ValueError(
            "No client models available."
        )

    print(
        f"[SERVER] Aggregating "
        f"{len(client_state_dicts)} client models..."
    )

    first_state = client_state_dicts[0]

    averaged = {}

    # --------------------------------------------------------
    # Average parameters
    # --------------------------------------------------------

    for key in first_state.keys():

        tensors = []

        valid = True

        for state in client_state_dicts:

            if key not in state:

                valid = False

                break

            tensor = state[key]

            if not torch.is_tensor(tensor):

                valid = False

                break

            tensors.append(
                tensor.float()
            )

        if not valid:

            print(
                f"[SERVER] Skipping key: {key}"
            )

            continue

        # ----------------------------------------------------
        # Floating point parameters
        # ----------------------------------------------------

        if tensors[0].dtype.is_floating_point:

            averaged[key] = torch.stack(
                tensors,
                dim=0
            ).mean(dim=0)

        else:

            # ------------------------------------------------
            # Integer/bool buffers
            # Use first client value
            # ------------------------------------------------

            averaged[key] = tensors[0]

    print(
        "[SERVER] Aggregation completed."
    )

    return averaged


# ============================================================
# 19. TRAIN ONE CLIENT
# ============================================================

def train_one_client(
    client_id,
    client_count,
    round_idx,
    global_state
):

    print("\n")
    print("=" * 80)
    print(
        f"CLIENT {client_id}/{client_count}"
    )
    print(
        f"FEDERATED ROUND {round_idx}"
    )
    print("=" * 80)

    data_yaml = get_client_yaml(
        client_id
    )

    data = load_client_data(
        client_id
    )

    # --------------------------------------------------------
    # Hyperparameter optimization
    # --------------------------------------------------------

    best_cfg, best_score = (
        swarm_horse_optimization(
            client_id=client_id,
            global_state=global_state,
            client_count=client_count,
            round_idx=round_idx
        )
    )

    # --------------------------------------------------------
    # Save best parameters
    # --------------------------------------------------------

    yaml_path, json_path = save_best_params(
        client_count=client_count,
        client_id=client_id,
        round_idx=round_idx,
        best_cfg=best_cfg,
        best_score=best_score
    )

    print(
        f"[Client {client_id}] "
        f"Best params saved:"
    )

    print(
        yaml_path
    )

    # --------------------------------------------------------
    # Create new local model
    # --------------------------------------------------------

    local_model = create_model()

    # --------------------------------------------------------
    # Load global weights
    # --------------------------------------------------------

    load_result = load_weights_into_model(
        local_model,
        global_state
    )

    # --------------------------------------------------------
    # Number of classes
    # --------------------------------------------------------

    num_classes = len(
        data["names"]
    )

    try:

        local_model.model.nc = num_classes

        local_model.model.model[-1].nc = (
            num_classes
        )

        local_model.model.model[-1].no = (
            num_classes + 5
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------

    print(
        f"\n[Client {client_id}] "
        f"Starting local training..."
    )

    start_time = time.time()

    results = local_model.train(

        data=data_yaml,

        epochs=best_cfg["epochs"],

        imgsz=IMGSZ,

        batch=best_cfg["batch"],

        lr0=best_cfg["lr0"],

        momentum=best_cfg["momentum"],

        weight_decay=best_cfg["weight_decay"],

        warmup_epochs=best_cfg["warmup_epochs"],

        verbose=VERBOSE,

        mosaic=0
    )

    training_time = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    try:
        precision = float(
            results.box.mp
        )
    except:
        precision = 0.0

    try:
        recall = float(
            results.box.mr
        )
    except:
        recall = 0.0

    try:
        map50 = float(
            results.box.map50
        )
    except:
        map50 = 0.0

    try:
        map5095 = float(
            results.box.map
        )
    except:
        map5095 = 0.0

    # --------------------------------------------------------
    # Local state dict
    # --------------------------------------------------------

    local_state = extract_state_dict(
        local_model
    )

    # --------------------------------------------------------
    # Save local model
    # --------------------------------------------------------

    local_model_path = save_local_model(

        model=local_model,

        client_count=client_count,

        client_id=client_id,

        round_idx=round_idx
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    client_dir = os.path.join(
        CLIENT_ROOT,
        f"client_{client_id}"
    )

    train_dir = os.path.join(
        client_dir,
        str(data.get("train", "train")),
        "images"
    )

    val_dir = os.path.join(
        client_dir,
        str(data.get("val", "val")),
        "images"
    )

    train_label_dir = os.path.join(
        client_dir,
        str(data.get("train", "train")),
        "labels"
    )

    val_label_dir = os.path.join(
        client_dir,
        str(data.get("val", "val")),
        "labels"
    )

    train_images = count_images(
        train_dir
    )

    val_images = count_images(
        val_dir
    )

    train_instances = count_instances(
        train_label_dir
    )

    val_instances = count_instances(
        val_label_dir
    )

    # --------------------------------------------------------
    # Metrics dictionary
    # --------------------------------------------------------

    metrics = {

        "client_count":
            client_count,

        "client_id":
            client_id,

        "round":
            round_idx,

        "precision":
            precision,

        "recall":
            recall,

        "map50":
            map50,

        "map50_95":
            map5095,

        "best_score":
            best_score,

        "train_images":
            train_images,

        "train_instances":
            train_instances,

        "val_images":
            val_images,

        "val_instances":
            val_instances,

        "training_time_sec":
            round(
                training_time,
                2
            )
    }

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_dir = os.path.join(
        RESULTS_ROOT,
        f"{client_count}_CLIENTS",
        "metrics"
    )

    os.makedirs(
        metrics_dir,
        exist_ok=True
    )

    metrics_path = os.path.join(
        metrics_dir,
        f"client_{client_id}_round_{round_idx}.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\n")
    print(
        f"[Client {client_id}] "
        f"Training completed."
    )

    print(
        f"Precision : {precision:.6f}"
    )

    print(
        f"Recall    : {recall:.6f}"
    )

    print(
        f"mAP50     : {map50:.6f}"
    )

    print(
        f"mAP50-95  : {map5095:.6f}"
    )

    print(
        f"Time      : {training_time:.2f} sec"
    )

    print(
        f"Local model:"
    )

    print(
        local_model_path
    )

    return (
        local_state,
        metrics,
        best_cfg,
        best_score
    )


# ============================================================
# 20. SAVE CSV RESULT
# ============================================================

def append_csv_result(
    client_count,
    metrics
):

    csv_dir = os.path.join(
        RESULTS_ROOT,
        f"{client_count}_CLIENTS"
    )

    os.makedirs(
        csv_dir,
        exist_ok=True
    )

    csv_path = os.path.join(
        csv_dir,
        "all_client_results.csv"
    )

    fields = [
        "client_count",
        "client_id",
        "round",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "best_score",
        "train_images",
        "train_instances",
        "val_images",
        "val_instances",
        "training_time_sec"
    ]

    file_exists = os.path.exists(
        csv_path
    )

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow(
            {
                field: metrics.get(
                    field,
                    ""
                )
                for field in fields
            }
        )


# ============================================================
# 21. SAVE GLOBAL SUMMARY
# ============================================================

def save_global_summary(
    client_count,
    round_idx,
    global_model_path,
    client_metrics
):

    directory = os.path.join(
        RESULTS_ROOT,
        f"{client_count}_CLIENTS",
        "global_summary"
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    summary = {

        "client_count":
            client_count,

        "round":
            round_idx,

        "global_model":
            global_model_path,

        "num_clients":
            len(client_metrics),

        "clients":
            client_metrics
    }

    path = os.path.join(
        directory,
        f"round_{round_idx}_summary.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    return path


# ============================================================
# 22. RUN ONE FEDERATED EXPERIMENT
# ============================================================

def run_federated_experiment(
    client_count
):

    print("\n\n")

    print(
        "#" * 80
    )

    print(
        f"# START FEDERATED EXPERIMENT"
    )

    print(
        f"# NUMBER OF CLIENTS = {client_count}"
    )

    print(
        "#" * 80
    )

    # --------------------------------------------------------
    # Check clients
    # --------------------------------------------------------

    active_clients = []

    for client_id in range(
        1,
        client_count + 1
    ):

        yaml_path = get_client_yaml(
            client_id
        )

        if not os.path.exists(
            yaml_path
        ):

            raise FileNotFoundError(

                f"Client {client_id} "
                f"does not exist.\n"
                f"Expected: {yaml_path}"
            )

        active_clients.append(
            client_id
        )

    # --------------------------------------------------------
    # Experiment directory
    # --------------------------------------------------------

    experiment_dir = os.path.join(
        RESULTS_ROOT,
        f"{client_count}_CLIENTS"
    )

    os.makedirs(
        experiment_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create initial global model
    # --------------------------------------------------------

    print(
        "\n[SERVER] Creating initial global model..."
    )

    initial_model = create_model()

    initial_state = extract_state_dict(
        initial_model
    )

    global_state = initial_state

    initial_global_path = os.path.join(
        experiment_dir,
        "global_models",
        "global_model_round_0.pt"
    )

    save_state_dict(
        global_state,
        initial_global_path
    )

    print(
        f"[SERVER] Initial model saved:"
    )

    print(
        initial_global_path
    )

    # --------------------------------------------------------
    # ROUNDS
    # --------------------------------------------------------

    experiment_start = time.time()
    #original range 1, rounds +1
    for round_idx in range(
        1,
        ROUNDS + 1
    ):

        print("\n\n")

        print(
            "=" * 80
        )

        print(
            f"FEDERATED ROUND {round_idx}/{ROUNDS}"
        )

        print(
            f"NUMBER OF CLIENTS: {client_count}"
        )

        print(
            "=" * 80
        )

        round_start = time.time()

        client_states = []

        client_metrics = []

        # ----------------------------------------------------
        # CLIENTS RUN SEQUENTIALLY
        # ----------------------------------------------------

        for client_id in active_clients:

            try:

                (
                    local_state,
                    metrics,
                    best_cfg,
                    best_score
                ) = train_one_client(

                    client_id=client_id,

                    client_count=client_count,

                    round_idx=round_idx,

                    global_state=global_state
                )

                # --------------------------------------------
                # Client finished
                # --------------------------------------------

                client_states.append(
                    local_state
                )

                client_metrics.append(
                    metrics
                )

                append_csv_result(
                    client_count,
                    metrics
                )

                print(
                    f"\n[SERVER] "
                    f"Received Client {client_id} "
                    f"Round {round_idx}"
                )

                print(
                    f"[SERVER] "
                    f"{len(client_states)}/"
                    f"{client_count} clients completed."
                )

            except Exception as e:

                print(
                    f"\n[ERROR] Client {client_id} "
                    f"failed."
                )

                print(
                    str(e)
                )

                traceback.print_exc()

                raise

        # ----------------------------------------------------
        # SERVER AGGREGATION
        # ----------------------------------------------------

        print("\n")

        print(
            "=" * 80
        )

        print(
            f"[SERVER] AGGREGATING ROUND {round_idx}"
        )

        print(
            f"[SERVER] Received "
            f"{len(client_states)}/{client_count} models"
        )

        print(
            "=" * 80
        )

        global_state = average_weights(
            client_states
        )

        # ----------------------------------------------------
        # Save global model
        # ----------------------------------------------------

        global_model_path = save_global_model(

            state_dict=global_state,

            client_count=client_count,

            round_idx=round_idx
        )

        checksum = file_sha256(
            global_model_path
        )

        print(
            f"\n[SERVER] Global model saved:"
        )

        print(
            global_model_path
        )

        print(
            f"[SERVER] SHA256:"
        )

        print(
            checksum
        )

        # ----------------------------------------------------
        # Save round summary
        # ----------------------------------------------------

        summary_path = save_global_summary(

            client_count=client_count,

            round_idx=round_idx,

            global_model_path=global_model_path,

            client_metrics=client_metrics
        )

        # ----------------------------------------------------
        # Round finished
        # ----------------------------------------------------

        round_time = (
            time.time()
            - round_start
        )

        print("\n")

        print(
            "=" * 80
        )

        print(
            f"ROUND {round_idx} FINISHED"
        )

        print(
            f"Time: {round_time:.2f} sec"
        )

        print(
            f"Global model:"
        )

        print(
            global_model_path
        )

        print(
            "=" * 80
        )

    # --------------------------------------------------------
    # Experiment completed
    # --------------------------------------------------------

    total_time = (
        time.time()
        - experiment_start
    )

    print("\n\n")

    print(
        "#" * 80
    )

    print(
        f"# EXPERIMENT COMPLETED"
    )

    print(
        f"# CLIENTS = {client_count}"
    )

    print(
        f"# ROUNDS = {ROUNDS}"
    )

    print(
        f"# TOTAL TIME = {total_time:.2f} sec"
    )

    print(
        "#" * 80
    )

    return {
        "client_count": client_count,
        "rounds": ROUNDS,
        "total_time_sec": total_time,
        "final_global_model": os.path.join(
            RESULTS_ROOT,
            f"{client_count}_CLIENTS",
            "global_models",
            f"global_model_round_{ROUNDS}.pt"
        )
    }


# ============================================================
# 23. MAIN EXPERIMENT LOOP
# ============================================================

all_experiments = []

overall_start = time.time()

for client_count in CLIENT_COUNTS:

    try:

        result = run_federated_experiment(
            client_count
        )

        all_experiments.append(
            result
        )

    except Exception as e:

        print("\n")

        print(
            "!" * 80
        )

        print(
            f"EXPERIMENT WITH "
            f"{client_count} CLIENTS FAILED"
        )

        print(
            str(e)
        )

        traceback.print_exc()

        print(
            "!" * 80
        )

        # ----------------------------------------------------
        # Continue with next client count
        # ----------------------------------------------------

        continue


# ============================================================
# 24. SAVE OVERALL SUMMARY
# ============================================================

overall_time = (
    time.time()
    - overall_start
)

overall_summary = {

    "client_counts":
        CLIENT_COUNTS,

    "rounds":
        ROUNDS,

    "pop_size":
        POP_SIZE,

    "max_iter":
        MAX_ITER,

    "imgsz":
        IMGSZ,

    "overall_time_sec":
        overall_time,

    "experiments":
        all_experiments
}

overall_path = os.path.join(
    RESULTS_ROOT,
    "overall_summary.json"
)

with open(
    overall_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        overall_summary,
        f,
        indent=4
    )


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n\n")

print(
    "=" * 80
)

print(
    "ALL FEDERATED EXPERIMENTS FINISHED"
)

print(
    "=" * 80
)

for result in all_experiments:

    print(
        f"Clients: "
        f"{result['client_count']:2d} | "
        f"Rounds: "
        f"{result['rounds']} | "
        f"Time: "
        f"{result['total_time_sec']:.2f} sec"
    )

print(
    "=" * 80
)

print(
    f"Overall time: "
    f"{overall_time:.2f} sec"
)

print(
    f"Overall summary:"
)

print(
    overall_path
)

print(
    "=" * 80
)


# ============================================================
# 26. PRINT RESULT DIRECTORY STRUCTURE
# ============================================================

print("\n")
print("RESULT DIRECTORY STRUCTURE")
print("=" * 80)

for client_count in CLIENT_COUNTS:

    base = os.path.join(
        RESULTS_ROOT,
        f"{client_count}_CLIENTS"
    )

    if os.path.exists(base):

        print(
            f"\n{base}/"
        )

        print(
            f"    global_models/"
        )

        print(
            f"        global_model_round_0.pt"
        )

        for round_idx in range(
            1,
            ROUNDS + 1
        ):

            print(
                f"        global_model_round_{round_idx}.pt"
            )

        print(
            f"    round_1/"
        )

        print(
            f"        client_1/"
        )

        print(
            f"            best_params.yaml"
        )

        print(
            f"            best_params.json"
        )

        print(
            f"        ..."
        )

        print(
            f"        client_{client_count}/"
        )

        print(
            f"            best_params.yaml"
        )

        print(
            f"            best_params.json"
        )

        print(
            f"    metrics/"
        )

        print(
            f"    all_client_results.csv"
        )

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)

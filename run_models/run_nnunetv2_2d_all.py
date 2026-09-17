from pathlib import Path
import json
import shutil
import subprocess
import sys

import pandas as pd

##########################################################
# Run order:

# 1. preprocess\prepare_nnunetv2_datasets.py
# 2. preprocess\check_nnunet_masks.py
# 3. run_models\run_nnunetv2_2d_all.py
# 4. run_models\predict_nnunetv2_2d_all.py
# 5. src\runners\evaluate_nnunetv2_predictions.py
# 6. run_models\visualize_nnunetv2_2d_all.py
##########################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.nnunet_env import (
    get_nnunet_environment,
    NNUNET_PREPROCESSED,
    NNUNET_RAW,
)


DATASETS = [
    # {
    #     "dataset_id": 501,
    #     "dataset_name": "ENID",
    #     "dataset_folder": "Dataset501_ENID",
    # },
    # {
    #     "dataset_id": 502,
    #     "dataset_name": "GLENDA",
    #     "dataset_folder": "Dataset502_GLENDA",
    # },
    # {
    #     "dataset_id": 503,
    #     "dataset_name": "GLENDA_clean",
    #     "dataset_folder": "Dataset503_GLENDA_clean",
    # },
    {
        "dataset_id": 504,
        "dataset_name": "GynSurg",
        "dataset_folder": "Dataset504_GynSurg",
    },
]

CONFIGURATION = "2d"
FOLD = "0"
TRAINER = "nnUNetTrainer_100epochs"


# Debug: run only GLENDA_clean
# DATASETS = [
#     {
#         "dataset_id": 503,
#         "dataset_name": "GLENDA_clean",
#         "dataset_folder": "Dataset503_GLENDA_clean",
#     },
# ]


def get_nnunet_executable(name: str) -> str:
    scripts_dir = Path(sys.executable).parent

    candidates = [
        scripts_dir / f"{name}.exe",
        scripts_dir / f"{name}.bat",
        scripts_dir / name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    found = shutil.which(name)

    if found is not None:
        return found

    raise FileNotFoundError(
        f"Could not find nnU-Net executable: {name}. "
        f"Expected it in: {scripts_dir}"
    )


def run_command(command, env):
    print("\n" + "=" * 100)
    print("Running command:")
    print(" ".join(command))
    print("=" * 100)

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        shell=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def verify_dataset_json_rgb(dataset_folder: str):
    dataset_json_path = NNUNET_RAW / dataset_folder / "dataset.json"

    if not dataset_json_path.exists():
        raise FileNotFoundError(f"dataset.json not found: {dataset_json_path}")

    with open(dataset_json_path, "r", encoding="utf-8") as file:
        dataset_json = json.load(file)

    channel_names = dataset_json.get("channel_names", {})
    num_channels = len(channel_names)

    print("\n" + "=" * 100)
    print(f"Checking dataset.json: {dataset_json_path}")
    print(f"channel_names: {channel_names}")
    print(f"num_channels: {num_channels}")
    print(f"reader/writer: {dataset_json.get('overwrite_image_reader_writer')}")
    print("=" * 100)

    if num_channels != 3:
        raise RuntimeError(
            "This dataset is still configured as non-RGB. "
            f"Expected 3 channels but got {num_channels}. "
            f"Fix this file: {dataset_json_path}"
        )

    if dataset_json.get("overwrite_image_reader_writer") != "NaturalImage2DIO":
        raise RuntimeError(
            "dataset.json must use NaturalImage2DIO for PNG images. "
            f"Fix this file: {dataset_json_path}"
        )


def load_conversion_report(dataset_name: str, dataset_folder: str) -> pd.DataFrame:
    report_path = (
        NNUNET_RAW
        / dataset_folder
        / f"{dataset_name}_nnunet_conversion_report.csv"
    )

    if not report_path.exists():
        raise FileNotFoundError(f"Conversion report not found: {report_path}")

    return pd.read_csv(report_path)


def rewrite_custom_split(dataset_name: str, dataset_folder: str):
    """
    Force nnU-Net fold 0 to use:
      train = original train split
      val   = original val split
    """

    conversion_df = load_conversion_report(
        dataset_name=dataset_name,
        dataset_folder=dataset_folder,
    )

    train_ids = conversion_df[conversion_df["split"] == "train"]["case_id"].tolist()
    val_ids = conversion_df[conversion_df["split"] == "val"]["case_id"].tolist()

    if len(train_ids) == 0:
        raise ValueError(f"No train cases found for {dataset_folder}")

    if len(val_ids) == 0:
        raise ValueError(f"No val cases found for {dataset_folder}")

    preprocessed_dataset_dir = NNUNET_PREPROCESSED / dataset_folder
    preprocessed_dataset_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        {
            "train": train_ids,
            "val": val_ids,
        }
    ]

    split_path = preprocessed_dataset_dir / "splits_final.json"

    with open(split_path, "w", encoding="utf-8") as file:
        json.dump(splits, file, indent=4)

    print(f"Custom split written: {split_path}")
    print(f"Train cases: {len(train_ids)}")
    print(f"Val cases:   {len(val_ids)}")


def plan_and_preprocess_dataset(dataset_id: int, env):
    plan_exe = get_nnunet_executable("nnUNetv2_plan_and_preprocess")

    run_command(
        [
            plan_exe,
            "-d",
            str(dataset_id),
            "-c",
            CONFIGURATION,
            "--verify_dataset_integrity",
        ],
        env=env,
    )


def train_dataset(dataset_id: int, env):
    train_exe = get_nnunet_executable("nnUNetv2_train")

    run_command(
        [
            train_exe,
            str(dataset_id),
            CONFIGURATION,
            FOLD,
            "-tr",
            TRAINER,
        ],
        env=env,
    )


def main():
    env = get_nnunet_environment()

    print("=" * 100)
    print(f"Planning and preprocessing nnU-Net v2 datasets")
    print(f"Trainer for training: {TRAINER}")
    print("=" * 100)

    for dataset_cfg in DATASETS:
        verify_dataset_json_rgb(dataset_cfg["dataset_folder"])

        plan_and_preprocess_dataset(
            dataset_id=dataset_cfg["dataset_id"],
            env=env,
        )

        rewrite_custom_split(
            dataset_name=dataset_cfg["dataset_name"],
            dataset_folder=dataset_cfg["dataset_folder"],
        )

    print("=" * 100)
    print(f"Training nnU-Net v2 2D fold 0 with {TRAINER}")
    print("=" * 100)

    for dataset_cfg in DATASETS:
        train_dataset(
            dataset_id=dataset_cfg["dataset_id"],
            env=env,
        )

    print("\nAll selected nnU-Net v2 2D 100-epoch trainings finished.")


if __name__ == "__main__":
    main()
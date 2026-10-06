from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.nnunet_env import (
    get_nnunet_environment,
    NNUNET_RAW,
    NNUNET_EXPORTS,
)


DATASETS = [
    # {
    #     "dataset_id": 501,
    #     "dataset_folder": "Dataset501_ENID",
    # },
    # {
    #     "dataset_id": 502,
    #     "dataset_folder": "Dataset502_GLENDA",
    # },
    # {
    #     "dataset_id": 503,
    #     "dataset_folder": "Dataset503_GLENDA_clean",
    # },
     {
             "dataset_id": 504,
            "dataset_folder": "Dataset504_GynSurg",
     },
]

CONFIGURATION = "2d"
FOLD = "0"
TRAINER = "nnUNetTrainer_100epochs"
CHECKPOINT = "checkpoint_best.pth"


# Debug: run only GLENDA_clean
# DATASETS = [
#     {
#         "dataset_id": 503,
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


def predict_dataset(dataset_id: int, dataset_folder: str, env):
    predict_exe = get_nnunet_executable("nnUNetv2_predict")

    images_tr = NNUNET_RAW / dataset_folder / "imagesTr"
    images_ts = NNUNET_RAW / dataset_folder / "imagesTs"

    output_all_images_tr = (
        NNUNET_EXPORTS
        / dataset_folder
        / "all_imagesTr_predictions_100ep"
    )

    output_test = (
        NNUNET_EXPORTS
        / dataset_folder
        / "test_predictions_100ep"
    )

    output_all_images_tr.mkdir(parents=True, exist_ok=True)
    output_test.mkdir(parents=True, exist_ok=True)

    if not images_tr.exists():
        raise FileNotFoundError(f"imagesTr not found: {images_tr}")

    if not images_ts.exists():
        raise FileNotFoundError(f"imagesTs not found: {images_ts}")

    run_command(
        [
            predict_exe,
            "-i",
            str(images_tr),
            "-o",
            str(output_all_images_tr),
            "-d",
            str(dataset_id),
            "-c",
            CONFIGURATION,
            "-f",
            FOLD,
            "-tr",
            TRAINER,
            "-chk",
            CHECKPOINT,
        ],
        env=env,
    )

    run_command(
        [
            predict_exe,
            "-i",
            str(images_ts),
            "-o",
            str(output_test),
            "-d",
            str(dataset_id),
            "-c",
            CONFIGURATION,
            "-f",
            FOLD,
            "-tr",
            TRAINER,
            "-chk",
            CHECKPOINT,
        ],
        env=env,
    )


def main():
    env = get_nnunet_environment()

    print("=" * 100)
    print(f"Predicting validation/test sets with nn-U-Net v2 2D")
    print(f"Trainer: {TRAINER}")
    print("=" * 100)

    for dataset_cfg in DATASETS:
        predict_dataset(
            dataset_id=dataset_cfg["dataset_id"],
            dataset_folder=dataset_cfg["dataset_folder"],
            env=env,
        )

    print("\nAll selected nnU-Net v2 100-epoch predictions finished.")


if __name__ == "__main__":
    main()
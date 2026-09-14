from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runners.run_deeplabv3plus import train_and_evaluate
from src.evaluation.visualize_metrics import visualize_dataset_model_split

# uses baseline venv

RESULTS_ROOT = Path(r"F:\Results\SAM_Benchmarking")

CONFIGS_TO_RUN = [
    #PROJECT_ROOT / "configs" / "experiments" / "enid_deeplabv3plus_trained.yaml",
    #PROJECT_ROOT / "configs" / "experiments" / "glenda_deeplabv3plus_trained.yaml",
    #PROJECT_ROOT / "configs" / "experiments" / "glenda_clean_deeplabv3plus_trained.yaml",
    PROJECT_ROOT / "configs" / "experiments" / "GynSurg_deeplabv3plus_trained.yaml",
]

# Debug: run only ENID
# CONFIGS_TO_RUN = [
#     PROJECT_ROOT / "configs" / "experiments" / "enid_deeplabv3plus_trained.yaml",
# ]

# Debug: run only original GLENDA
# CONFIGS_TO_RUN = [
#     PROJECT_ROOT / "configs" / "experiments" / "glenda_deeplabv3plus_trained.yaml",
# ]

# Debug: run only cleaned GLENDA
# CONFIGS_TO_RUN = [
#     PROJECT_ROOT / "configs" / "experiments" / "glenda_clean_deeplabv3plus_trained.yaml",
# ]

DATASETS_TO_VISUALIZE = [
    #"ENID",
    #"GLENDA",
    #"GLENDA_clean",
    "GynSurg"
]

SPLITS_TO_VISUALIZE = [
    "val",
    "test",
]


def main():
    print("=" * 100)
    print("Running DeepLabV3Plus baseline experiments")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 100)

    for config_path in CONFIGS_TO_RUN:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print("\n" + "=" * 100)
        print(f"Running config: {config_path}")
        print("=" * 100)

        train_and_evaluate(str(config_path))

    print("\nAll selected DeepLabV3Plus experiments finished.")

    print("\n" + "=" * 100)
    print("Visualizing DeepLabV3Plus metrics")
    print("=" * 100)

    for dataset_name in DATASETS_TO_VISUALIZE:
        for split in SPLITS_TO_VISUALIZE:
            try:
                visualize_dataset_model_split(
                    results_root=RESULTS_ROOT,
                    dataset_name=dataset_name,
                    model_name="DeepLabV3Plus",
                    training_state="trained",
                    split=split,
                )
            except Exception as error:
                print(
                    f"WARNING: Visualization failed for "
                    f"{dataset_name} | DeepLabV3Plus | {split}: {error}"
                )

    print("\nAll selected DeepLabV3Plus experiments and visualizations finished.")


if __name__ == "__main__":
    main()
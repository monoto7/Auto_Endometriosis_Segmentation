from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runners.run_yolo_seg import train_and_evaluate
from src.evaluation.visualize_metrics import visualize_dataset_model_split


RESULTS_ROOT = Path(r"C:\Users\cooll\OneDrive\Documents\SPARC\\OrganSegmentationTesting\\")

CONFIGS_TO_RUN = [
    #PROJECT_ROOT / "configs" / "experiments" / "enid_yolo11s_seg_trained.yaml",
    #PROJECT_ROOT / "configs" / "experiments" / "glenda_yolo11s_seg_trained.yaml",
    #PROJECT_ROOT / "configs" / "experiments" / "glenda_clean_yolo11s_seg_trained.yaml",
    PROJECT_ROOT / "configs" / "experiments" / "GynSurg_yolo11s_seg.yaml",
]

# Debug: run only ENID
# CONFIGS_TO_RUN = [
#     PROJECT_ROOT / "configs" / "experiments" / "enid_yolo11s_seg_trained.yaml",
# ]

# Debug: run only original GLENDA
# CONFIGS_TO_RUN = [
#     PROJECT_ROOT / "configs" / "experiments" / "glenda_yolo11s_seg_trained.yaml",
# ]

# Debug: run only cleaned GLENDA
# CONFIGS_TO_RUN = [
#     PROJECT_ROOT / "configs" / "experiments" / "glenda_clean_yolo11s_seg_trained.yaml",
# ]

DATASETS_TO_VISUALIZE = [
    #"ENID",
    #"GLENDA",
    #"GLENDA_clean",
    "GynSurg",
]

SPLITS_TO_VISUALIZE = [
    "val",
    "test",
]


def main():
    print("=" * 100)
    print("Running YOLO11s-seg baseline experiments")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 100)

    for config_path in CONFIGS_TO_RUN:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print("\n" + "=" * 100)
        print(f"Running config: {config_path}")
        print("=" * 100)

        train_and_evaluate(str(config_path))

    print("\nAll selected YOLO11s-seg experiments finished.")

    print("\n" + "=" * 100)
    print("Visualizing YOLO11s-seg metrics")
    print("=" * 100)

    for dataset_name in DATASETS_TO_VISUALIZE:
        for split in SPLITS_TO_VISUALIZE:
            try:
                visualize_dataset_model_split(
                    results_root=RESULTS_ROOT,
                    dataset_name=dataset_name,
                    model_name="YOLO11s_seg",
                    training_state="trained",
                    split=split,
                )
            except Exception as error:
                print(
                    f"WARNING: Visualization failed for "
                    f"{dataset_name} | YOLO11s_seg | {split}: {error}"
                )

    print("\nAll selected YOLO11s-seg experiments and visualizations finished.")


if __name__ == "__main__":
    main()
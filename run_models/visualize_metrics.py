from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.visualize_metrics import visualize_dataset_model_split


# ============================================================
# Generic metric visualization launcher
# ============================================================

RESULTS_ROOT = Path(r"C:\Users\cooll\OneDrive\Documents\SPARC\\OrganSegmentationTesting\\")

# Change these when visualizing another model.
MODEL_NAME = "SAM2"
TRAINING_STATE = ""

# Comment out anything you do not want.
DATASETS_TO_VISUALIZE = [
    #"ENID",
    #"GLENDA",
    "GynSurg",
    #"Fluoro",
]

SPLITS_TO_VISUALIZE = [
    "val",
    "test",
]


def main():
    for dataset_name in DATASETS_TO_VISUALIZE:
        for split in SPLITS_TO_VISUALIZE:
            visualize_dataset_model_split(
                results_root=RESULTS_ROOT,
                dataset_name=dataset_name,
                model_name=MODEL_NAME,
                training_state=TRAINING_STATE,
                split=split,
            )


if __name__ == "__main__":
    main()
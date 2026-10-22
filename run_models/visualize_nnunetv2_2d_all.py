from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.visualize_metrics import visualize_dataset_model_split

from src.utils.nnunet_env import (
    NNUNET_EPOCH_STRING
)

RESULTS_ROOT = Path(r"/data/m_meyer/AES/OrganSegmentationResults/")

DATASETS_TO_VISUALIZE = [
    #"ENID",
    #"GLENDA",
    #"GLENDA_clean",
    #"GynSurg",
    #"GynSurg85",
    #"GynSurg170",
    "GynSurg255",
]

SPLITS_TO_VISUALIZE = [
    "val",
    "test",
]


def main():
    for dataset_name in DATASETS_TO_VISUALIZE:
        for split in SPLITS_TO_VISUALIZE:
            try:
                visualize_dataset_model_split(
                    results_root=RESULTS_ROOT,
                    dataset_name=dataset_name,
                    model_name=f"nnUNetV2_2D{NNUNET_EPOCH_STRING}",
                    training_state="trained",
                    split=split,
                )
            except Exception as error:
                print(
                    f"WARNING: Visualization failed for "
                    f"{dataset_name} | nnUNetV2_2D | {split}: {error}"
                )


if __name__ == "__main__":
    main()
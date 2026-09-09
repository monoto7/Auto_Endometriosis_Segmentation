from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runners.run_medsam_oracle import run_experiment
from src.evaluation.visualize_metrics import visualize_dataset_model_split


# ============================================================
# Run frozen MedSAM oracle-box experiments
# ============================================================

RESULTS_ROOT = Path(r"C:\Users\cooll\OneDrive\Documents\SPARC\\OrganSegmentationTesting\\")


CONFIGS_TO_RUN = [
    #PROJECT_ROOT / "configs" / "experiments" / "enid_surgisam2_frozen_oracle.yaml",
    #PROJECT_ROOT / "configs" / "experiments" / "glenda_surgisam2_frozen_oracle.yaml",
    #PROJECT_ROOT / "configs" / "experiments" / "glenda_clean_surgisam2_frozen_oracle.yaml",
    PROJECT_ROOT / "configs" / "experiments" / "GynSurg_MedSAM.yaml",
    #PROJECT_ROOT / "configs" / "experiments" / "Fluoro_medsam.yaml",
]

# Run only ENID:
# CONFIGS_TO_RUN = [
#     PROJECT_ROOT / "configs" / "experiments" / "enid_medsam_frozen_oracle.yaml",
# ]

# Run only GLENDA:
# CONFIGS_TO_RUN = [
#     PROJECT_ROOT / "configs" / "experiments" / "glenda_medsam_frozen_oracle.yaml",
# ]

DATASETS_TO_VISUALIZE = [
    #"ENID",
    #"GLENDA",
    #"GLENDA_clean",
    "GynSurg",
    #Fluoro",
]

SPLITS_TO_VISUALIZE = [
    #"val",
    #"test",
    "train"
]


def main():
    print("=" * 100)
    print("Running MedSAM frozen oracle-box experiments")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 100)

    for config_path in CONFIGS_TO_RUN:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print("\n" + "=" * 100)
        print(f"Running config: {config_path}")
        print("=" * 100)

        run_experiment(str(config_path))

    print("\nAll selected MedSAM experiments finished.")

    print("\n" + "=" * 100)
    print("Visualizing MedSAM metrics")
    print("=" * 100)

    for dataset_name in DATASETS_TO_VISUALIZE:
        for split in SPLITS_TO_VISUALIZE:
            try:
                visualize_dataset_model_split(
                    results_root=RESULTS_ROOT,
                    dataset_name=dataset_name,
                    model_name="MedSAM",
                    training_state="",
                    split=split,
                )
            except Exception as error:
                print(
                    f"WARNING: Visualization failed for "
                    f"{dataset_name} | {split}: {error}"
                )

    print("\nAll selected MedSAM experiments and metric visualizations finished.")


if __name__ == "__main__":
    main()
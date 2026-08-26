from pathlib import Path
import sys


# ============================================================
# Run frozen SAM2 oracle-prompt experiments
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runners.run_sam2_oracle import run_experiment


# ------------------------------------------------------------
# Choose which datasets to run.
# Comment out the one you do not want.
# ------------------------------------------------------------

CONFIGS_TO_RUN = [
    #PROJECT_ROOT / "configs" / "experiments" / "enid_sam2_frozen_oracle.yaml",
    #PROJECT_ROOT / "configs" / "experiments" / "glenda_sam2_frozen_oracle.yaml",
    #PROJECT_ROOT / "configs" / "experiments" / "glenda_clean_sam2_frozen_oracle.yaml",
    #PROJECT_ROOT / "configs" / "experiments" / "GynSurg_sam2.yaml",
    PROJECT_ROOT / "configs" / "experiments" / "Fluoro_sam2.yaml",
]

# Example: run only ENID
# CONFIGS_TO_RUN = [
#     PROJECT_ROOT / "configs" / "experiments" / "enid_sam2_frozen_oracle.yaml",
# ]

# Example: run only GLENDA
# CONFIGS_TO_RUN = [
#     PROJECT_ROOT / "configs" / "experiments" / "glenda_sam2_frozen_oracle.yaml",
# ]


def main():
    print("=" * 100)
    print("Running SAM2 frozen oracle-prompt experiments")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 100)

    for config_path in CONFIGS_TO_RUN:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print("\n" + "=" * 100)
        print(f"Running config: {config_path}")
        print("=" * 100)

        run_experiment(str(config_path))

    print("\nAll selected SAM2 experiments finished.")


if __name__ == "__main__":
    main()
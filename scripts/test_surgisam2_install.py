from pathlib import Path
import os
import sys
import torch


SAM2_ROOT = Path(r"C:\Users\cooll\OneDrive\Documents\SPARC\Models\SAM2\sam2")
SURGISAM2_CHECKPOINT = Path(
    r"C:\Users\cooll\OneDrive\Documents\SPARC\SAM checkpoints\Curated400_checkpoint_26.pt"
)
# Start with original SAM2 Base Plus config.
MODEL_CFG = r"C:\Users\cooll\OneDrive\Documents\SPARC\Models\SAM2\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_b+.yaml"


def main():
    print("=" * 100)
    print("Testing SurgiSAM2 checkpoint loading")
    print("=" * 100)

    print(f"Python executable: {sys.executable}")
    print(f"Torch version:     {torch.__version__}")
    print(f"CUDA available:    {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:               {torch.cuda.get_device_name(0)}")

    if not SAM2_ROOT.exists():
        raise FileNotFoundError(f"SAM2 repo not found: {SAM2_ROOT}")

    if not SURGISAM2_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SurgiSAM2 checkpoint not found: {SURGISAM2_CHECKPOINT}"
        )

    print(f"SAM2 root:         {SAM2_ROOT}")
    print(f"Checkpoint:        {SURGISAM2_CHECKPOINT}")
    print(f"Model cfg:         {MODEL_CFG}")

    sys.path.insert(0, str(SAM2_ROOT))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    old_cwd = Path.cwd()

    try:
        os.chdir(SAM2_ROOT)

        model = build_sam2(
            config_file=MODEL_CFG,
            ckpt_path=str(SURGISAM2_CHECKPOINT),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        predictor = SAM2ImagePredictor(model)

    finally:
        os.chdir(old_cwd)

    print("=" * 100)
    print("SurgiSAM2 checkpoint loaded successfully.")
    print(f"Predictor type: {type(predictor)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
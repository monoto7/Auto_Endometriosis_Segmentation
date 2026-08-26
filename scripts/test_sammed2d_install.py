from pathlib import Path
import sys
import torch


SAMMED2D_ROOT = Path(r"C:\Users\cooll\OneDrive\Documents\SPARC\Models\\SAM-Med2d\\SAM-Med2D")
CHECKPOINT = Path(r"C:\Users\cooll\OneDrive\Documents\SPARC\SAM checkpoints\sam-med2d_b.pth")


def main():
    print("=" * 100)
    print("Testing SAM-Med2D checkpoint loading")
    print("=" * 100)

    print(f"Python executable: {sys.executable}")
    print(f"Torch version:     {torch.__version__}")
    print(f"CUDA available:    {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:               {torch.cuda.get_device_name(0)}")

    if not SAMMED2D_ROOT.exists():
        raise FileNotFoundError(f"SAM-Med2D repo not found: {SAMMED2D_ROOT}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM-Med2D checkpoint not found: {CHECKPOINT}")

    repo_root_str = str(SAMMED2D_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from segment_anything import sam_model_registry

    class Args:
        model_type = "vit_b"
        image_size = 256
        sam_checkpoint = str(CHECKPOINT)
        encoder_adapter = True

    args = Args()

    model = sam_model_registry["vit_b"](args)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print("=" * 100)
    print("SAM-Med2D checkpoint loaded successfully.")
    print(f"Model type: {type(model)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
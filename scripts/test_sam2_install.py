import sys
from pathlib import Path
import torch

SAM2_ROOT = Path(r"C:\Users\cooll\OneDrive\Documents\SPARC\\Models\\SAM2\\sam2")
CHECKPOINT = r"C:\Users\cooll\OneDrive\Documents\SPARC\SAM checkpoints\\sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

sys.path.insert(0, str(SAM2_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

model = build_sam2(
    config_file=MODEL_CFG,
    ckpt_path=CHECKPOINT,
    device="cpu",
)

predictor = SAM2ImagePredictor(model)

print("SAM2 loaded successfully.")
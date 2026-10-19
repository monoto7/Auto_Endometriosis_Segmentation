from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def save_overlay(
    image_path: Path,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    output_path: Path,
    alpha: float = 0.45,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    overlay = image_np.copy()

    gt = gt_mask > 0
    pred = pred_mask > 0

    # GT = blue
    overlay[gt] = [0, 0, 255]

    # Prediction = vermillion
    overlay[pred] = [227, 66, 52]

    # Overlap = Blue tinted green
    overlay[gt & pred] = [0, 158, 115]

    blended = cv2.addWeighted(image_np, 1 - alpha, overlay, alpha, 0)
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_path), blended_bgr)
from pathlib import Path
import numpy as np
from PIL import Image
import cv2


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    return (arr > 0).astype(np.uint8)


#No longer converting mask into binary, as we want to handle multiple classes and the data preparation should already do this for single class situations
def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    return mask_np


def save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask_uint8, mode="L").save(output_path)


def merge_binary_masks(masks):
    if len(masks) == 0:
        return None

    merged = np.zeros_like(masks[0], dtype=np.uint8)

    for mask in masks:
        merged = np.logical_or(merged > 0, mask > 0).astype(np.uint8)

    return merged


def find_image_path(images_dir: Path, image_name: str) -> Path:
    candidate = images_dir / image_name
    if candidate.exists():
        return candidate

    stem = Path(image_name).stem

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find image for {image_name} in {images_dir}")


def empty_mask_like_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    return np.zeros((height, width), dtype=np.uint8)


def ensure_same_size(mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    """
    Safety check. SAM2 should return original-size masks, but this prevents crashes
    if any backend returns a different resolution.
    """
    if mask.shape == reference_mask.shape:
        return mask

    from PIL import Image

    resized = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
        (reference_mask.shape[1], reference_mask.shape[0]),
        resample=Image.NEAREST,
    )

    return (np.array(resized) > 0).astype(np.uint8)

#new funcs

def probability_to_mask(probability_map, threshold, postprocessing_cfg, classes):
    mask = np.zeros(probability_map[0].shape)
    for i in range(probability_map.shape[0]-1):
        mask[probability_map[i+1] >= threshold] = int(classes[i])

        #TODO fix below so it works with multiclass
        if postprocessing_cfg.get("remove_small_components", False):
            min_area_px = int(postprocessing_cfg.get("min_component_area_px", 0))
            mask = remove_small_components(mask, min_area_px,classes[i])

    return mask

def remove_small_components(original_mask: np.ndarray, min_area_px: int, class_id = "255"):
    if min_area_px <= 0:
        return original_mask

    mask = original_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area_px:
            cleaned[labels == label_id] = 1

    return cleaned.astype(np.uint8) * int(class_id)
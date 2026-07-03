from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = mask[..., 0]

    return (mask > 0).astype(np.uint8)


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-8) -> float:
    a = binarize_mask(mask_a).astype(bool)
    b = binarize_mask(mask_b).astype(bool)

    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()

    if denom == 0:
        return 1.0

    return float((2.0 * inter + eps) / (denom + eps))


def iou_score(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-8) -> float:
    a = binarize_mask(mask_a).astype(bool)
    b = binarize_mask(mask_b).astype(bool)

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 1.0

    return float((inter + eps) / (union + eps))


def precision_score(mask_pred: np.ndarray, mask_gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = binarize_mask(mask_pred).astype(bool)
    gt = binarize_mask(mask_gt).astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()

    if tp + fp == 0:
        return 1.0 if gt.sum() == 0 else 0.0

    return float((tp + eps) / (tp + fp + eps))


def recall_score(mask_pred: np.ndarray, mask_gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = binarize_mask(mask_pred).astype(bool)
    gt = binarize_mask(mask_gt).astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    if tp + fn == 0:
        return 1.0

    return float((tp + eps) / (tp + fn + eps))


def get_connected_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    binary_mask = binarize_mask(binary_mask)
    labeled, n_components = ndimage.label(binary_mask)
    return labeled, int(n_components)


def get_component_mask(labeled_mask: np.ndarray, component_id: int) -> np.ndarray:
    return (labeled_mask == component_id).astype(np.uint8)


def get_tight_box(mask: np.ndarray) -> Optional[List[int]]:
    mask = binarize_mask(mask)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    return [x1, y1, x2, y2]


def clip_point(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))
    return x, y


def get_positive_point_from_component(component_mask: np.ndarray) -> Tuple[int, int]:
    component_mask = binarize_mask(component_mask)
    ys, xs = np.where(component_mask > 0)

    if len(xs) == 0:
        h, w = component_mask.shape
        return w // 2, h // 2

    cx = float(xs.mean())
    cy = float(ys.mean())

    distances = (xs - cx) ** 2 + (ys - cy) ** 2
    idx = int(np.argmin(distances))

    return int(xs[idx]), int(ys[idx])


def choose_background_point(
    candidate_mask: np.ndarray,
    target_x: int,
    target_y: int,
) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(candidate_mask > 0)

    if len(xs) == 0:
        return None

    distances = (xs - target_x) ** 2 + (ys - target_y) ** 2
    idx = int(np.argmin(distances))

    return int(xs[idx]), int(ys[idx])


def get_negative_points_box_background_or_fallback(
    component_mask: np.ndarray,
    box_xyxy: List[int],
    fallback_offset_px: int = 5,
) -> List[Tuple[int, int]]:
    """
    Creates four negative points:
    1. top background point
    2. bottom background point
    3. left background point
    4. right background point

    Preferred source:
    background inside tight box = box area minus SegFormer component.

    Fallback:
    if the box is too tight and no background exists in a side region,
    place the point 5 px outside the box in that direction.
    The box itself is not expanded.
    """
    component_mask = binarize_mask(component_mask)
    height, width = component_mask.shape

    x1, y1, x2, y2 = [int(v) for v in box_xyxy]

    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))

    box_mask = np.zeros_like(component_mask, dtype=np.uint8)
    box_mask[y1:y2 + 1, x1:x2 + 1] = 1

    box_background = ((box_mask > 0) & (component_mask == 0)).astype(np.uint8)

    top_candidates = np.zeros_like(component_mask, dtype=np.uint8)
    top_candidates[y1:cy + 1, x1:x2 + 1] = box_background[y1:cy + 1, x1:x2 + 1]

    bottom_candidates = np.zeros_like(component_mask, dtype=np.uint8)
    bottom_candidates[cy:y2 + 1, x1:x2 + 1] = box_background[cy:y2 + 1, x1:x2 + 1]

    left_candidates = np.zeros_like(component_mask, dtype=np.uint8)
    left_candidates[y1:y2 + 1, x1:cx + 1] = box_background[y1:y2 + 1, x1:cx + 1]

    right_candidates = np.zeros_like(component_mask, dtype=np.uint8)
    right_candidates[y1:y2 + 1, cx:x2 + 1] = box_background[y1:y2 + 1, cx:x2 + 1]

    top_point = choose_background_point(top_candidates, cx, y1)
    bottom_point = choose_background_point(bottom_candidates, cx, y2)
    left_point = choose_background_point(left_candidates, x1, cy)
    right_point = choose_background_point(right_candidates, x2, cy)

    if top_point is None:
        top_point = clip_point(cx, y1 - fallback_offset_px, width, height)

    if bottom_point is None:
        bottom_point = clip_point(cx, y2 + fallback_offset_px, width, height)

    if left_point is None:
        left_point = clip_point(x1 - fallback_offset_px, cy, width, height)

    if right_point is None:
        right_point = clip_point(x2 + fallback_offset_px, cy, width, height)

    return [top_point, bottom_point, left_point, right_point]


def build_box_posneg_prompt_from_component(
    component_mask: np.ndarray,
    fallback_offset_px: int = 5,
) -> Optional[Dict]:
    component_mask = binarize_mask(component_mask)

    if component_mask.sum() == 0:
        return None

    box_xyxy = get_tight_box(component_mask)

    if box_xyxy is None:
        return None

    positive_point = get_positive_point_from_component(component_mask)

    negative_points = get_negative_points_box_background_or_fallback(
        component_mask=component_mask,
        box_xyxy=box_xyxy,
        fallback_offset_px=fallback_offset_px,
    )

    point_coords = [positive_point] + negative_points
    point_labels = [1, 0, 0, 0, 0]

    return {
        "box_xyxy": box_xyxy,
        "positive_point": positive_point,
        "negative_points": negative_points,
        "point_coords": point_coords,
        "point_labels": point_labels,
    }


def select_best_candidate_by_segformer_dice(
    candidate_masks: np.ndarray,
    segformer_component_mask: np.ndarray,
) -> Dict:
    segformer_component_mask = binarize_mask(segformer_component_mask)

    candidate_masks = np.asarray(candidate_masks)

    if candidate_masks.ndim == 2:
        candidate_masks = candidate_masks[None, ...]

    best_index = 0
    best_dice = -1.0
    best_iou = -1.0
    best_mask = None

    for idx in range(candidate_masks.shape[0]):
        candidate = binarize_mask(candidate_masks[idx])

        dsc = dice_score(candidate, segformer_component_mask)
        iou = iou_score(candidate, segformer_component_mask)

        if dsc > best_dice:
            best_index = int(idx)
            best_dice = float(dsc)
            best_iou = float(iou)
            best_mask = candidate

    return {
        "best_index": best_index,
        "best_mask": best_mask.astype(np.uint8),
        "dice_agreement": best_dice,
        "iou_agreement": best_iou,
    }


def apply_dice_fallback(
    surgisam_mask: np.ndarray,
    segformer_component_mask: np.ndarray,
    dice_agreement: float,
    dice_threshold: float = 0.80,
) -> Dict:
    """
    Requirement:
    fallback to SegFormer mask if Dice <= 0.80.
    """
    surgisam_mask = binarize_mask(surgisam_mask)
    segformer_component_mask = binarize_mask(segformer_component_mask)

    if dice_agreement > dice_threshold:
        return {
            "final_mask": surgisam_mask.astype(np.uint8),
            "accepted_surgisam2": True,
            "fallback_used": False,
            "fallback_reason": "accepted_surgisam2_dice_above_0p80",
        }

    return {
        "final_mask": segformer_component_mask.astype(np.uint8),
        "accepted_surgisam2": False,
        "fallback_used": True,
        "fallback_reason": "fallback_to_segformer_dice_lte_0p80",
    }


def points_to_string(points: List[Tuple[int, int]]) -> str:
    return ";".join([f"{int(x)},{int(y)}" for x, y in points])


def box_to_string(box_xyxy: List[int]) -> str:
    return ",".join([str(int(v)) for v in box_xyxy])
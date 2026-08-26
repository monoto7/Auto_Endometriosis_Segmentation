from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# Multiclass Oracle prompt generation for GynSurg
# Positive point + tight box + local negative background points
# ============================================================

DATASETS = {
    "GynSurg": Path(r"C:\\Users\\cooll\\OneDrive\\Documents\\SPARC\\SplitDatasets\\GynSurg")
}

SPLITS = ["train", "val", "test"]

# Ignore tiny foreground specks.
MIN_COMPONENT_AREA = 10

# Negative-point settings.
NUM_NEGATIVE_POINTS = 4

# First expand the tight box by this margin to find local background.
INITIAL_NEGATIVE_MARGIN_PX = 10

# If not enough background exists, progressively expand up to this margin.
MAX_NEGATIVE_MARGIN_PX = 80

# Expansion step.
NEGATIVE_MARGIN_STEP_PX = 10

# Save quality-control images with boxes and points.
SAVE_VISUAL_CHECKS = False


def load_grayscale_masks(mask_path: Path):
    """
    Loads mask as:
      0 = background
      1 = foreground
    """
    
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    uniques = np.unique(mask_np[mask_np>0])
    masks = []
    for val in uniques:
        masks.append((mask_np == val).astype(np.uint8))
    return masks,uniques


def find_connected_components(binary_mask: np.ndarray):
    """
    Finds separate lesion/pathology components.

    Returns one component per disconnected foreground region.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    components = []
    lesion_id = 0

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < MIN_COMPONENT_AREA:
            continue

        component_mask = (labels == label_id).astype(np.uint8)

        bbox = [
            x,
            y,
            x + w - 1,
            y + h - 1
        ]

        components.append({
            "lesion_id": lesion_id,
            "component_mask": component_mask,
            "area": area,
            "bbox": bbox,
        })

        lesion_id += 1

    return components


def get_center_positive_point(component_mask: np.ndarray):
    """
    Gets a central point inside the lesion.

    Uses distance transform, so the point is inside the foreground and
    far from the boundary when possible.
    """
    component_mask = component_mask.astype(np.uint8)

    distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(distance)

    if max_val > 0:
        x, y = max_loc
        return [int(x), int(y)]

    ys, xs = np.where(component_mask > 0)

    if len(xs) == 0:
        return None

    return [int(np.mean(xs)), int(np.mean(ys))]


def clip_box(box, width, height):
    x1, y1, x2, y2 = box

    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))

    return [x1, y1, x2, y2]


def expand_box(box, margin, width, height):
    x1, y1, x2, y2 = box

    return clip_box(
        [x1 - margin, y1 - margin, x2 + margin, y2 + margin],
        width,
        height
    )


def nearest_candidate_to_target(candidate_points_xy: np.ndarray, target_xy):
    """
    candidate_points_xy shape: [N, 2], columns x,y.
    target_xy: [x,y]
    """
    if candidate_points_xy.shape[0] == 0:
        return None

    target = np.array(target_xy, dtype=np.float32)

    distances = np.sum((candidate_points_xy.astype(np.float32) - target) ** 2, axis=1)
    idx = int(np.argmin(distances))

    return candidate_points_xy[idx].astype(int).tolist()


def get_background_candidates_in_region(binary_mask, region_box):
    """
    Returns background pixels inside region_box.

    Uses the full binary mask, not only the component mask.
    Therefore negative points cannot be placed inside neighboring lesions.
    """
    h, w = binary_mask.shape
    x1, y1, x2, y2 = clip_box(region_box, w, h)

    roi = binary_mask[y1:y2 + 1, x1:x2 + 1]

    ys, xs = np.where(roi == 0)

    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    xs = xs + x1
    ys = ys + y1

    return np.stack([xs, ys], axis=1).astype(np.int32)


def directional_candidates(candidate_points_xy, tight_box, direction):
    """
    Filters background candidate points by direction relative to the tight box.

    direction:
      top, bottom, left, right
    """
    x1, y1, x2, y2 = tight_box

    xs = candidate_points_xy[:, 0]
    ys = candidate_points_xy[:, 1]

    if direction == "top":
        keep = ys < y1
    elif direction == "bottom":
        keep = ys > y2
    elif direction == "left":
        keep = xs < x1
    elif direction == "right":
        keep = xs > x2
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return candidate_points_xy[keep]


def generate_negative_points(binary_mask: np.ndarray, tight_box):
    """
    Generates 4 local negative points:
      top, bottom, left, right

    Main rule:
      negative region = expanded tight box - full binary mask

    Fallback:
      If a direction has no background candidate, increase expansion margin.
      Points are always required to be on global background.
    """
    h, w = binary_mask.shape
    x1, y1, x2, y2 = tight_box

    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))

    direction_targets = {
        "top": [cx, y1],
        "bottom": [cx, y2],
        "left": [x1, cy],
        "right": [x2, cy],
    }

    negative_points = []

    for direction in ["top", "bottom", "left", "right"]:
        selected_point = None

        for margin in range(
            INITIAL_NEGATIVE_MARGIN_PX,
            MAX_NEGATIVE_MARGIN_PX + 1,
            NEGATIVE_MARGIN_STEP_PX
        ):
            expanded = expand_box(tight_box, margin, w, h)

            candidates = get_background_candidates_in_region(
                binary_mask=binary_mask,
                region_box=expanded
            )

            if candidates.shape[0] == 0:
                continue

            directional = directional_candidates(
                candidate_points_xy=candidates,
                tight_box=tight_box,
                direction=direction
            )

            if directional.shape[0] == 0:
                continue

            selected_point = nearest_candidate_to_target(
                candidate_points_xy=directional,
                target_xy=direction_targets[direction]
            )

            if selected_point is not None:
                break

        if selected_point is not None:
            if selected_point not in negative_points:
                negative_points.append(selected_point)

    # Final fallback:
    # If fewer than 4 points were found directionally, fill using nearest
    # background pixels around the expanded box, still avoiding all foreground.
    if len(negative_points) < NUM_NEGATIVE_POINTS:
        expanded = expand_box(tight_box, MAX_NEGATIVE_MARGIN_PX, w, h)
        candidates = get_background_candidates_in_region(
            binary_mask=binary_mask,
            region_box=expanded
        )

        if candidates.shape[0] > 0:
            fallback_targets = [
                [cx, max(0, y1 - MAX_NEGATIVE_MARGIN_PX)],
                [cx, min(h - 1, y2 + MAX_NEGATIVE_MARGIN_PX)],
                [max(0, x1 - MAX_NEGATIVE_MARGIN_PX), cy],
                [min(w - 1, x2 + MAX_NEGATIVE_MARGIN_PX), cy],
                [max(0, x1 - MAX_NEGATIVE_MARGIN_PX), max(0, y1 - MAX_NEGATIVE_MARGIN_PX)],
                [min(w - 1, x2 + MAX_NEGATIVE_MARGIN_PX), max(0, y1 - MAX_NEGATIVE_MARGIN_PX)],
                [max(0, x1 - MAX_NEGATIVE_MARGIN_PX), min(h - 1, y2 + MAX_NEGATIVE_MARGIN_PX)],
                [min(w - 1, x2 + MAX_NEGATIVE_MARGIN_PX), min(h - 1, y2 + MAX_NEGATIVE_MARGIN_PX)],
            ]

            for target in fallback_targets:
                if len(negative_points) >= NUM_NEGATIVE_POINTS:
                    break

                point = nearest_candidate_to_target(candidates, target)

                if point is not None and point not in negative_points:
                    negative_points.append(point)

    # Safety check: only keep points that are truly background.
    safe_points = []

    for x, y in negative_points:
        if 0 <= y < h and 0 <= x < w and binary_mask[y, x] == 0:
            if [x, y] not in safe_points:
                safe_points.append([x, y])

    return safe_points[:NUM_NEGATIVE_POINTS]


def draw_visual_check(image_path, mask_path, prompt_rows, out_path):
    """
    Saves visual check:
      green = tight box
      red = positive point
      blue = negative points
    """
    image = Image.open(image_path).convert("RGB")
    vis = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for row in prompt_rows:
        x1 = int(row["bbox_x1"])
        y1 = int(row["bbox_y1"])
        x2 = int(row["bbox_x2"])
        y2 = int(row["bbox_y2"])

        pos = json.loads(row["positive_points_xy"])
        negs = json.loads(row["negative_points_xy"])

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for p in pos:
            px, py = p
            cv2.circle(vis, (int(px), int(py)), 5, (0, 0, 255), -1)

        for n in negs:
            nx, ny = n
            cv2.circle(vis, (int(nx), int(ny)), 5, (255, 0, 0), -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def process_split(dataset_name: str, dataset_root: Path, split: str):
    images_dir = dataset_root / split / "images"
    masks_dir = dataset_root / split / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    prompt_dir = dataset_root / "oracle_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    visual_dir = dataset_root / "oracle_prompt_visual_checks" / split

    image_paths = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.jpeg")) +
        list(images_dir.glob("*.png"))
    )

    all_rows = []
    missing_masks = []
    empty_masks = 0
    total_components = 0
    components_without_4_negative_points = 0

    for image_path in image_paths:
        mask_path = masks_dir / f"{image_path.stem}.png"

        if not mask_path.exists():
            missing_masks.append(image_path.name)
            continue

        masks,grayscale_values = load_grayscale_masks(mask_path)
        if masks.count==1 and masks[0].sum() == 0:
            empty_masks += 1
        else:
            #Setting to -1 to it iterates to 0 at the start of the loop, to have mask_id logic near the top
            mask = -1
            for binary_mask in masks:
                mask+=1
                if binary_mask.sum() == 0:
                    continue

                components = find_connected_components(binary_mask)

                if len(components) == 0:
                    empty_masks += 1
                    continue

                image_prompt_rows = []

                for component in components:
                    lesion_id = component["lesion_id"]
                    component_mask = component["component_mask"]
                    area = component["area"]
                    bbox = component["bbox"]

                    positive_point = get_center_positive_point(component_mask)
                    negative_points = generate_negative_points(binary_mask, bbox)

                    if len(negative_points) < NUM_NEGATIVE_POINTS:
                        components_without_4_negative_points += 1

                    x1, y1, x2, y2 = bbox

                    point_coords = []
                    point_labels = []

                    if positive_point is not None:
                        point_coords.append(positive_point)
                        point_labels.append(1)

                    for neg in negative_points:
                        point_coords.append(neg)
                        point_labels.append(0)

                    row = {
                        "dataset": dataset_name,
                        "split": split,

                        "image_name": image_path.name,
                        "mask_name": mask_path.name,

                        #set the class_id as the grayscale_value of the class
                        "class_id": grayscale_values[mask],
                        "lesion_id": lesion_id,
                        "component_area_px": area,

                        # Tight GT box around this lesion component.
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "bbox_xyxy": json.dumps([x1, y1, x2, y2]),

                        # Positive point.
                        "positive_point_x": positive_point[0] if positive_point else None,
                        "positive_point_y": positive_point[1] if positive_point else None,
                        "positive_points_xy": json.dumps([positive_point] if positive_point else []),

                        # Local negative background points.
                        "negative_points_xy": json.dumps(negative_points),
                        "num_negative_points": len(negative_points),

                        # Directly usable for SAM APIs.
                        "point_coords_xy": json.dumps(point_coords),
                        "point_labels": json.dumps(point_labels),

                        # Prompt mode labels.
                        "prompt_gt_point": json.dumps([positive_point] if positive_point else []),
                        "prompt_gt_box": json.dumps([x1, y1, x2, y2]),
                        "prompt_gt_box_plus_point": json.dumps({
                            "box": [x1, y1, x2, y2],
                            "points": [positive_point] if positive_point else [],
                            "labels": [1] if positive_point else []
                        }),
                        "prompt_gt_box_plus_pos_neg_points": json.dumps({
                            "box": [x1, y1, x2, y2],
                            "points": point_coords,
                            "labels": point_labels
                        }),
                    }

                    all_rows.append(row)
                    image_prompt_rows.append(row)
                    total_components += 1

            if SAVE_VISUAL_CHECKS:
                out_vis_path = visual_dir / f"{image_path.stem}_oracle_prompts.jpg"
                draw_visual_check(
                    image_path=image_path,
                    mask_path=mask_path,
                    prompt_rows=image_prompt_rows,
                    out_path=out_vis_path
                )

    df = pd.DataFrame(all_rows)

    out_csv = prompt_dir / f"{dataset_name}_{split}_oracle_prompts.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{dataset_name} - {split}")
    print(f"  images found:                         {len(image_paths)}")
    print(f"  missing masks:                        {len(missing_masks)}")
    print(f"  empty masks skipped:                  {empty_masks}")
    print(f"  lesion components / prompt rows:      {total_components}")
    print(f"  components with <4 negative points:   {components_without_4_negative_points}")
    print(f"  saved CSV:                            {out_csv}")

    if missing_masks:
        print("  first missing masks:")
        for name in missing_masks[:10]:
            print(f"    {name}")

    return df


def combine_dataset_csvs(dataset_name: str, dataset_root: Path):
    prompt_dir = dataset_root / "oracle_prompts"

    dfs = []

    for split in SPLITS:
        csv_path = prompt_dir / f"{dataset_name}_{split}_oracle_prompts.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)

    if not dfs:
        return

    combined = pd.concat(dfs, ignore_index=True)
    out_csv = prompt_dir / f"{dataset_name}_all_oracle_prompts.csv"
    combined.to_csv(out_csv, index=False)

    print(f"\n{dataset_name} combined prompt CSV:")
    print(f"  saved: {out_csv}")
    print(f"  rows:  {len(combined)}")


def main():
    for dataset_name, dataset_root in DATASETS.items():
        print("\n" + "=" * 80)
        print(f"Generating oracle prompts for {dataset_name}")
        print("=" * 80)

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        for split in SPLITS:
            process_split(dataset_name, dataset_root, split)

        combine_dataset_csvs(dataset_name, dataset_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
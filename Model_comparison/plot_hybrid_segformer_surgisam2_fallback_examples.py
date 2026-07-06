from pathlib import Path
import sys
import json
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Paths
# =============================================================================

RESULTS_ROOT = Path(r"F:\Results\SAM_Benchmarking")

OUTPUT_ROOT = (
    RESULTS_ROOT
    / "Model_comparison"
    / "Hybrid_SegFormer_SurgiSAM2_Fallback_Inference_comparison"
)

STANDARDIZED_DATASETS = {
    "ENID": Path(r"F:\Datasets\Standardized datasets\ENID\ENID 60_20_20 Split"),
    "GLENDA": Path(r"F:\Datasets\Standardized datasets\GLENDA\GLENDA 60_20_20 split"),
    "GLENDA_clean": Path(r"F:\Datasets\Standardized datasets\GLENDA_clean\GLENDA_clean 60_20_20 split"),
}


# =============================================================================
# Tuned fallback hybrid result folder
# =============================================================================

HYBRID_METHOD_FOLDER = "SegFormer_SurgiSAM2_AutoBox_Fallback"
HYBRID_TRAINING_STATE = "hybrid"

# IMPORTANT:
# This must match the tuned fallback runner output folder.
HYBRID_PROMPT_MODE = "Auto_box_fallback_dice_0p85_area_0p70_1p30"


# =============================================================================
# Examples
# =============================================================================
# root_name = original image filename without extension.
# split should usually be "test" or "val".

EXAMPLES = {
    "ENID": {
        "split": "test",
        #"root_name": "c_1_v_(video_7.mp4)_f_0",
        #"root_name": "c_253_v_(video_8049.mp4)_f_147",
        #"root_name": "c_232_v_(video_7308.mp4)_f_0",
        "root_name": "c_3_v_(video_24.mp4)_f_137",
        #"root_name": "c_157_v_(video_4727.mp4)_f_0",
    },
    "GLENDA": {
        "split": "test",
        #"root_name": "c_141_v_(video_4254.mp4)_f_5",
        ###"root_name": "c_86_v_(video_2617.mp4)_f_177",
        #"root_name": "c_141_v_(video_4253.mp4)_f_55",
        "root_name": "c_125_v_(video_3726.mp4)_f_247",
        ###"root_name": "c_62_v_(video_2045.mp4)_f_363",
    },
    "GLENDA_clean": {
        "split": "test",
        #"root_name": "c_141_v_(video_4254.mp4)_f_5",
        #"root_name": "c_86_v_(video_2617.mp4)_f_177",
        #"root_name": "c_141_v_(video_4253.mp4)_f_55",
        "root_name": "c_125_v_(video_3726.mp4)_f_247",
        #"root_name": "c_62_v_(video_2045.mp4)_f_363",
    },
}


# =============================================================================
# Plot settings
# =============================================================================

IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
]

DPI = 600

TITLE_FONT_SIZE = 20
DICE_FONT_SIZE = 20

PANEL_WIDTH = 3.8
PANEL_HEIGHT = 5.6

OVERLAY_ALPHA = 0.55

GT_ONLY_COLOR = np.array([0, 255, 0], dtype=np.float32)       # green
PRED_ONLY_COLOR = np.array([255, 0, 0], dtype=np.float32)     # red
OVERLAP_COLOR = np.array([255, 255, 0], dtype=np.float32)     # yellow
BOX_COLOR = (180, 0, 255)                                    # purple

BOX_WIDTH = 4


# =============================================================================
# Basic helpers
# =============================================================================

def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 18,
        }
    )


def get_nearest_resampling():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.NEAREST
    return Image.NEAREST


def find_file_by_root(folder: Path, root_name: str):
    for extension in IMAGE_EXTENSIONS:
        candidate = folder / f"{root_name}{extension}"
        if candidate.exists():
            return candidate

    matches = []

    for extension in IMAGE_EXTENSIONS:
        matches.extend(sorted(folder.glob(f"{root_name}*{extension}")))

    if matches:
        return matches[0]

    return None


def read_rgb_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def read_binary_mask(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    return (mask_np > 0).astype(np.uint8)


def resize_mask_to_image(mask_np: np.ndarray, image_shape):
    target_height, target_width = image_shape[:2]

    if mask_np.shape[:2] == (target_height, target_width):
        return (mask_np > 0).astype(np.uint8)

    mask_image = Image.fromarray((mask_np > 0).astype(np.uint8) * 255)
    mask_image = mask_image.resize(
        (target_width, target_height),
        resample=get_nearest_resampling(),
    )

    return (np.array(mask_image) > 0).astype(np.uint8)


def make_gt_display(mask_np: np.ndarray):
    return (mask_np > 0).astype(np.uint8) * 255


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    if pred_mask.shape != gt_mask.shape:
        pred_mask = resize_mask_to_image(pred_mask, gt_mask.shape)

    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0

    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, ~gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()

    denominator = 2 * tp + fp + fn

    if denominator == 0:
        return 1.0

    return float((2 * tp) / denominator)


def format_dice(dice_value):
    if dice_value is None:
        return "Dice: n/a"

    try:
        if pd.isna(dice_value):
            return "Dice: n/a"
    except Exception:
        pass

    return f"Dice: {float(dice_value):.3f}"


# =============================================================================
# Overlay helpers
# =============================================================================

def draw_boxes(
    image_np: np.ndarray,
    boxes_xyxy,
    box_color=BOX_COLOR,
    box_width=BOX_WIDTH,
):
    output = Image.fromarray(image_np.astype(np.uint8))
    draw = ImageDraw.Draw(output)

    if boxes_xyxy is None:
        boxes_xyxy = []

    for box in boxes_xyxy:
        if box is None or len(box) != 4:
            continue

        x_min, y_min, x_max, y_max = [int(v) for v in box]

        for offset in range(box_width):
            draw.rectangle(
                [
                    x_min - offset,
                    y_min - offset,
                    x_max + offset,
                    y_max + offset,
                ],
                outline=tuple(box_color),
            )

    return np.array(output)


def make_segmentation_overlay(
    image_np: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    boxes_xyxy=None,
    draw_prompt_boxes: bool = False,
):
    """
    Green  = GT only / missed lesion
    Red    = prediction only / false positive
    Yellow = overlap / true positive
    Purple boxes optional
    """

    gt_mask = resize_mask_to_image(gt_mask, image_np.shape)
    pred_mask = resize_mask_to_image(pred_mask, image_np.shape)

    gt_bool = gt_mask > 0
    pred_bool = pred_mask > 0

    overlap_bool = gt_bool & pred_bool
    gt_only_bool = gt_bool & (~pred_bool)
    pred_only_bool = pred_bool & (~gt_bool)

    image_float = image_np.astype(np.float32)
    overlay = image_float.copy()

    overlay[gt_only_bool] = (
        (1.0 - OVERLAY_ALPHA) * image_float[gt_only_bool]
        + OVERLAY_ALPHA * GT_ONLY_COLOR
    )

    overlay[pred_only_bool] = (
        (1.0 - OVERLAY_ALPHA) * image_float[pred_only_bool]
        + OVERLAY_ALPHA * PRED_ONLY_COLOR
    )

    overlay[overlap_bool] = (
        (1.0 - OVERLAY_ALPHA) * image_float[overlap_bool]
        + OVERLAY_ALPHA * OVERLAP_COLOR
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if draw_prompt_boxes:
        overlay = draw_boxes(
            image_np=overlay,
            boxes_xyxy=boxes_xyxy,
            box_color=BOX_COLOR,
            box_width=BOX_WIDTH,
        )

    return overlay


def make_segformer_binary_display(segformer_mask: np.ndarray, boxes_xyxy):
    """
    Shows SegFormer binary prediction in white on black,
    with purple prompt boxes overlaid.
    """

    mask_display = (segformer_mask > 0).astype(np.uint8) * 255
    rgb_display = np.stack([mask_display, mask_display, mask_display], axis=-1)

    rgb_display = draw_boxes(
        image_np=rgb_display,
        boxes_xyxy=boxes_xyxy,
        box_color=BOX_COLOR,
        box_width=BOX_WIDTH,
    )

    return rgb_display


# =============================================================================
# Fallback hybrid-output loading
# =============================================================================

def get_hybrid_split_dir(dataset_key: str, split: str):
    return (
        RESULTS_ROOT
        / dataset_key
        / HYBRID_METHOD_FOLDER
        / HYBRID_TRAINING_STATE
        / HYBRID_PROMPT_MODE
        / split
    )


def find_source_image_and_mask(dataset_key: str, split: str, root_name: str):
    dataset_root = STANDARDIZED_DATASETS[dataset_key]

    image_folder = dataset_root / split / "images"
    mask_folder = dataset_root / split / "masks"

    image_path = find_file_by_root(image_folder, root_name)
    mask_path = find_file_by_root(mask_folder, root_name)

    if image_path is None:
        raise FileNotFoundError(
            f"Could not find source image for root '{root_name}' in {image_folder}"
        )

    if mask_path is None:
        raise FileNotFoundError(
            f"Could not find source mask for root '{root_name}' in {mask_folder}"
        )

    return image_path, mask_path


def find_hybrid_masks(dataset_key: str, split: str, root_name: str):
    split_dir = get_hybrid_split_dir(dataset_key, split)

    segformer_mask_dir = split_dir / "segformer_initial_masks"
    final_mask_dir = split_dir / "merged_masks"

    segformer_mask_path = find_file_by_root(segformer_mask_dir, root_name)
    final_mask_path = find_file_by_root(final_mask_dir, root_name)

    if segformer_mask_path is None:
        raise FileNotFoundError(
            f"Could not find SegFormer initial mask for '{root_name}' in {segformer_mask_dir}"
        )

    if final_mask_path is None:
        raise FileNotFoundError(
            f"Could not find final tuned fallback hybrid mask for '{root_name}' in {final_mask_dir}"
        )

    return segformer_mask_path, final_mask_path


def parse_boxes_from_value(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, float) and pd.isna(value):
        return []

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return []


def find_row_for_root(metrics_df: pd.DataFrame, root_name: str):
    candidate_columns = [
        "case_id",
        "image_name",
        "mask_name",
        "prediction_name",
        "merged_mask_name",
    ]

    for column in candidate_columns:
        if column not in metrics_df.columns:
            continue

        stems = metrics_df[column].astype(str).apply(lambda x: Path(x).stem)
        matches = metrics_df[stems == root_name]

        if len(matches) > 0:
            return matches.iloc[0]

    for column in candidate_columns:
        if column not in metrics_df.columns:
            continue

        matches = metrics_df[
            metrics_df[column].astype(str).str.contains(root_name, case=False, regex=False)
        ]

        if len(matches) > 0:
            return matches.iloc[0]

    return None


def load_hybrid_metrics_row(dataset_key: str, split: str, root_name: str):
    split_dir = get_hybrid_split_dir(dataset_key, split)
    metrics_path = split_dir / "metrics_image_level.csv"

    if not metrics_path.exists():
        return None, metrics_path

    metrics_df = pd.read_csv(metrics_path)
    row = find_row_for_root(metrics_df, root_name)

    return row, metrics_path


def load_prompt_json(dataset_key: str, split: str, root_name: str):
    split_dir = get_hybrid_split_dir(dataset_key, split)
    prompt_dir = split_dir / "auto_prompts"

    prompt_path = prompt_dir / f"{root_name}.json"

    if not prompt_path.exists():
        matches = sorted(prompt_dir.glob(f"{root_name}*.json"))

        if matches:
            prompt_path = matches[0]
        else:
            return {}, None

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt_data = json.load(file)

    return prompt_data, prompt_path


def load_component_decisions(dataset_key: str, split: str, root_name: str):
    split_dir = get_hybrid_split_dir(dataset_key, split)
    decision_dir = split_dir / "component_decisions"

    decision_path = decision_dir / f"{root_name}.json"

    if not decision_path.exists():
        matches = sorted(decision_dir.glob(f"{root_name}*.json"))

        if matches:
            decision_path = matches[0]
        else:
            return {}, None

    with open(decision_path, "r", encoding="utf-8") as file:
        decision_data = json.load(file)

    return decision_data, decision_path


def load_boxes(dataset_key: str, split: str, root_name: str, metrics_row):
    prompt_data, prompt_path = load_prompt_json(dataset_key, split, root_name)

    if "boxes_xyxy" in prompt_data:
        boxes = prompt_data["boxes_xyxy"]
        return boxes, prompt_path

    if metrics_row is not None and "boxes_xyxy" in metrics_row.index:
        boxes = parse_boxes_from_value(metrics_row["boxes_xyxy"])
        return boxes, None

    return [], prompt_path


def get_metric_from_row(row, metric_name):
    if row is None:
        return None

    lower_to_original = {
        str(column).lower(): column
        for column in row.index
    }

    metric_name_lower = metric_name.lower()

    if metric_name_lower not in lower_to_original:
        return None

    value = row[lower_to_original[metric_name_lower]]

    try:
        return float(value)
    except Exception:
        return None


def get_value_from_row(row, column_name):
    if row is None:
        return None

    if column_name not in row.index:
        return None

    return row[column_name]


def get_component_summary(metrics_row, decision_data):
    accepted = None
    fallback = None
    rate = None

    if metrics_row is not None:
        accepted = get_value_from_row(metrics_row, "accepted_surgisam2_components")
        fallback = get_value_from_row(metrics_row, "fallback_to_segformer_components")
        rate = get_value_from_row(metrics_row, "acceptance_rate_components")

    if accepted is None and "accepted_surgisam2_components" in decision_data:
        accepted = decision_data["accepted_surgisam2_components"]

    if fallback is None and "fallback_to_segformer_components" in decision_data:
        fallback = decision_data["fallback_to_segformer_components"]

    if rate is None and "acceptance_rate_components" in decision_data:
        rate = decision_data["acceptance_rate_components"]

    try:
        accepted = int(accepted)
    except Exception:
        accepted = 0

    try:
        fallback = int(fallback)
    except Exception:
        fallback = 0

    try:
        rate = float(rate)
    except Exception:
        total = accepted + fallback
        rate = accepted / total if total > 0 else 0.0

    return accepted, fallback, rate


# =============================================================================
# Plotting
# =============================================================================

def plot_hybrid_fallback_example(dataset_key: str, split: str, root_name: str):
    print("\n" + "=" * 100)
    print(f"Creating tuned fallback hybrid qualitative figure: {dataset_key} | {split} | {root_name}")
    print("=" * 100)

    image_path, gt_path = find_source_image_and_mask(
        dataset_key=dataset_key,
        split=split,
        root_name=root_name,
    )

    segformer_mask_path, final_mask_path = find_hybrid_masks(
        dataset_key=dataset_key,
        split=split,
        root_name=root_name,
    )

    metrics_row, metrics_path = load_hybrid_metrics_row(
        dataset_key=dataset_key,
        split=split,
        root_name=root_name,
    )

    boxes_xyxy, prompt_path = load_boxes(
        dataset_key=dataset_key,
        split=split,
        root_name=root_name,
        metrics_row=metrics_row,
    )

    decision_data, decision_path = load_component_decisions(
        dataset_key=dataset_key,
        split=split,
        root_name=root_name,
    )

    accepted_count, fallback_count, acceptance_rate = get_component_summary(
        metrics_row=metrics_row,
        decision_data=decision_data,
    )

    image_np = read_rgb_image(image_path)
    gt_mask = read_binary_mask(gt_path)
    segformer_mask = read_binary_mask(segformer_mask_path)
    final_mask = read_binary_mask(final_mask_path)

    segformer_mask = resize_mask_to_image(segformer_mask, image_np.shape)
    final_mask = resize_mask_to_image(final_mask, image_np.shape)

    segformer_dice = get_metric_from_row(metrics_row, "segformer_initial_dice")
    final_dice = get_metric_from_row(metrics_row, "dice")

    if segformer_dice is None:
        segformer_dice = compute_dice(segformer_mask, gt_mask)

    if final_dice is None:
        final_dice = compute_dice(final_mask, gt_mask)

    segformer_initial_overlay = make_segmentation_overlay(
        image_np=image_np,
        gt_mask=gt_mask,
        pred_mask=segformer_mask,
        boxes_xyxy=None,
        draw_prompt_boxes=False,
    )

    segformer_binary_with_boxes = make_segformer_binary_display(
        segformer_mask=segformer_mask,
        boxes_xyxy=boxes_xyxy,
    )

    fallback_final_overlay = make_segmentation_overlay(
        image_np=image_np,
        gt_mask=gt_mask,
        pred_mask=final_mask,
        boxes_xyxy=boxes_xyxy,
        draw_prompt_boxes=True,
    )

    panels = [
        {
            "title": "Input Image",
            "bottom": "",
            "image": image_np,
            "mode": "rgb",
        },
        {
            "title": "GT",
            "bottom": "",
            "image": make_gt_display(gt_mask),
            "mode": "gray",
        },
        {
            "title": "SegFormer Initial",
            "bottom": format_dice(segformer_dice),
            "image": segformer_initial_overlay,
            "mode": "rgb",
        },
        {
            "title": "Auto Box Prompts",
            "bottom": f"Boxes: {len(boxes_xyxy)}",
            "image": segformer_binary_with_boxes,
            "mode": "rgb",
        },
        {
            "title": "Tuned Fallback Final",
            "bottom": (
                f"{format_dice(final_dice)} \n"
                f"SAM accepted: {accepted_count}, fallback: {fallback_count}"
            ),
            "image": fallback_final_overlay,
            "mode": "rgb",
        },
    ]

    n_panels = len(panels)

    fig_width = PANEL_WIDTH * n_panels
    fig_height = PANEL_HEIGHT

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    axes = axes[0]

    for axis, panel in zip(axes, panels):
        if panel["mode"] == "gray":
            axis.imshow(panel["image"], cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(panel["image"])

        axis.set_title(
            panel["title"],
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
            pad=7,
        )

        axis.text(
            0.5,
            -0.075,
            panel["bottom"],
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=DICE_FONT_SIZE,
        )

        axis.set_xticks([])
        axis.set_yticks([])

        for spine in axis.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(
        left=0.002,
        right=0.998,
        top=0.82,
        bottom=0.16,
        wspace=0.006,
        hspace=0.0,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output_png = (
        OUTPUT_ROOT
        / f"{dataset_key}_{split}_{root_name}_tuned_fallback_hybrid_segformer_surgisam2_5panel.png"
    )

    output_csv = (
        OUTPUT_ROOT
        / f"{dataset_key}_{split}_{root_name}_tuned_fallback_hybrid_segformer_surgisam2_5panel_files.csv"
    )

    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.015,
    )

    plt.close(fig)

    loaded_row = {
        "dataset": dataset_key,
        "split": split,
        "root_name": root_name,
        "method_folder": HYBRID_METHOD_FOLDER,
        "prompt_mode": HYBRID_PROMPT_MODE,
        "image_path": str(image_path),
        "gt_path": str(gt_path),
        "segformer_mask_path": str(segformer_mask_path),
        "final_tuned_fallback_hybrid_mask_path": str(final_mask_path),
        "metrics_path": str(metrics_path),
        "prompt_path": str(prompt_path) if prompt_path is not None else "",
        "component_decision_path": str(decision_path) if decision_path is not None else "",
        "num_boxes": len(boxes_xyxy),
        "accepted_surgisam2_components": accepted_count,
        "fallback_to_segformer_components": fallback_count,
        "acceptance_rate_components": acceptance_rate,
        "boxes_xyxy": json.dumps(boxes_xyxy),
        "segformer_initial_dice": segformer_dice,
        "tuned_fallback_final_dice": final_dice,
        "output_png": str(output_png),
    }

    pd.DataFrame([loaded_row]).to_csv(output_csv, index=False)

    print(f"Saved PNG: {output_png}")
    print(f"Saved CSV: {output_csv}")
    print(f"Number of purple boxes: {len(boxes_xyxy)}")
    print(f"Accepted SurgiSAM2 components: {accepted_count}")
    print(f"Fallback to SegFormer components: {fallback_count}")


def main():
    setup_matplotlib()

    for dataset_key, example_cfg in EXAMPLES.items():
        split = example_cfg["split"]
        root_name = example_cfg["root_name"]

        if root_name.startswith("PUT_"):
            print(f"Skipping {dataset_key}. Set a real root_name first.")
            continue

        plot_hybrid_fallback_example(
            dataset_key=dataset_key,
            split=split,
            root_name=root_name,
        )


if __name__ == "__main__":
    main()
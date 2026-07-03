from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_ROOT = Path(r"F:\Results\SAM_Benchmarking")
OUTPUT_ROOT = RESULTS_ROOT / "Model_comparison" / "Inference_comparison"/ "new plots"


STANDARDIZED_DATASETS = {
    "ENID": Path(r"F:\Datasets\Standardized datasets\ENID\ENID 60_20_20 Split"),
    "GLENDA": Path(r"F:\Datasets\Standardized datasets\GLENDA\GLENDA 60_20_20 split"),
    "GLENDA_clean": Path(r"F:\Datasets\Standardized datasets\GLENDA_clean\GLENDA_clean 60_20_20 split"),
}


# Change these :
# root_name = original image filename without extension.
# split should usually be "test" or "val".
EXAMPLES = {
    "ENID": {
        "split": "test",
        #"root_name": "c_1_v_(video_7.mp4)_f_0",
        #"root_name": "c_253_v_(video_8049.mp4)_f_147",
        #"root_name": "c_232_v_(video_7308.mp4)_f_0",
        #"root_name": "c_3_v_(video_24.mp4)_f_137",
        "root_name": "c_157_v_(video_4727.mp4)_f_0",

    },
    "GLENDA": {
        "split": "test",
        #"root_name": "c_141_v_(video_4254.mp4)_f_5",
        ####"root_name": "c_86_v_(video_2617.mp4)_f_177",
        #"root_name": "c_141_v_(video_4253.mp4)_f_55",
        "root_name": "c_125_v_(video_3726.mp4)_f_247",
        ###"root_name": "c_62_v_(video_2045.mp4)_f_363",
    },

    "GLENDA_clean": {
        "split": "test",
        #"root_name": "c_141_v_(video_4254.mp4)_f_5",
        #"root_name": "c_86_v_(video_2617.mp4)_f_177",
        #"root_name": "c_141_v_(video_4253.mp4)_f_55",
        #"root_name": "c_125_v_(video_3726.mp4)_f_247",
        "root_name": "c_62_v_(video_2045.mp4)_f_363",
    },
}


IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
]


SAM_PROMPT_MODE_TO_COMPARE = None

SAM_PROMPT_MODE_CANDIDATES = [
    "GT_box",
    "GT_box_point",
    "GT_point",
    "Box_prompt",
    "Point_prompt",
    "No_prompt",
    "Auto_YOLO_box",
]


MODEL_ORDER = [
    {
        "display_name": "SAM2",
        "folder_candidates": ["SAM2"],
        "training_state": "frozen",
        "prompt_mode": SAM_PROMPT_MODE_TO_COMPARE,
        "is_sam": True,
    },
    {
        "display_name": "MedSAM",
        "folder_candidates": ["MedSAM"],
        "training_state": "frozen",
        "prompt_mode": SAM_PROMPT_MODE_TO_COMPARE,
        "is_sam": True,
    },
    {
        "display_name": "SAM-Med2D",
        "folder_candidates": ["SAM-Med2D", "SAMMed2D", "SAM_Med2D"],
        "training_state": "frozen",
        "prompt_mode": SAM_PROMPT_MODE_TO_COMPARE,
        "is_sam": True,
    },
    {
        "display_name": "SurgiSAM2",
        "folder_candidates": ["SurgiSAM2"],
        "training_state": "frozen",
        "prompt_mode": SAM_PROMPT_MODE_TO_COMPARE,
        "is_sam": True,
    },
    {
        "display_name": "YOLO11s-seg",
        "folder_candidates": ["YOLO11s_seg"],
        "training_state": "trained",
        "prompt_mode": "No_prompt",
        "is_sam": False,
    },
    {
        "display_name": "DeepLabV3+",
        "folder_candidates": ["DeepLabV3Plus"],
        "training_state": "trained",
        "prompt_mode": "No_prompt",
        "is_sam": False,
    },
    {
        "display_name": "SegFormer",
        "folder_candidates": ["SegFormer"],
        "training_state": "trained",
        "prompt_mode": "No_prompt",
        "is_sam": False,
    },
    {
        "display_name": "UNet++",
        "folder_candidates": ["UNetPP"],
        "training_state": "trained",
        "prompt_mode": "No_prompt",
        "is_sam": False,
    },
    {
        "display_name": "nnU-Net v2 2D",
        "folder_candidates": ["nnUNetV2_2D_100ep", "nnUNetV2_2D"],
        "training_state": "trained",
        "prompt_mode": "No_prompt",
        "is_sam": False,
    },
]


DPI = 600

TITLE_FONT_SIZE = 20
DICE_FONT_SIZE = 20

# Increase these if you want even larger one-row panels.
PANEL_WIDTH = 2.80
PANEL_HEIGHT = 5.20
# PANEL_WIDTH = 0.5
# PANEL_HEIGHT = 3

GT_ONLY_COLOR = np.array([0, 255, 0], dtype=np.float32)       # green
PRED_ONLY_COLOR = np.array([255, 0, 0], dtype=np.float32)     # red
OVERLAP_COLOR = np.array([255, 255, 0], dtype=np.float32)     # yellow

OVERLAY_ALPHA = 0.55


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
        return mask_np

    mask_image = Image.fromarray((mask_np > 0).astype(np.uint8) * 255)
    mask_image = mask_image.resize(
        (target_width, target_height),
        resample=Image.NEAREST,
    )

    return (np.array(mask_image) > 0).astype(np.uint8)


def make_overlay(image_np: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Qualitative overlay:
    green  = GT only / missed lesion
    red    = prediction only / false positive
    yellow = GT and prediction overlap / true positive
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

    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_gt_display(mask_np: np.ndarray):
    return (mask_np > 0).astype(np.uint8) * 255


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


def find_metrics_file_for_model(dataset_key: str, split: str, model_spec: dict):
    for folder_name in model_spec["folder_candidates"]:
        model_root = (
            RESULTS_ROOT
            / dataset_key
            / folder_name
            / model_spec["training_state"]
        )

        if not model_root.exists():
            continue

        if model_spec["is_sam"]:
            if model_spec["prompt_mode"] is not None:
                prompt_candidates = [model_spec["prompt_mode"]]
            else:
                prompt_candidates = SAM_PROMPT_MODE_CANDIDATES

            for prompt_mode in prompt_candidates:
                candidate = model_root / prompt_mode / split / "metrics_image_level.csv"

                if candidate.exists():
                    return candidate, prompt_mode, folder_name

            fallback_candidates = sorted(
                model_root.glob(f"*/{split}/metrics_image_level.csv")
            )

            if fallback_candidates:
                path = fallback_candidates[0]
                prompt_mode = path.parents[1].name
                return path, prompt_mode, folder_name

        else:
            prompt_mode = model_spec["prompt_mode"]

            candidate = model_root / prompt_mode / split / "metrics_image_level.csv"

            if candidate.exists():
                return candidate, prompt_mode, folder_name

            fallback_candidates = sorted(
                model_root.glob(f"*/{split}/metrics_image_level.csv")
            )

            if fallback_candidates:
                path = fallback_candidates[0]
                prompt_mode = path.parents[1].name
                return path, prompt_mode, folder_name

    return None, None, None


def find_row_for_root(metrics_df: pd.DataFrame, root_name: str):
    root_name = str(root_name)

    candidate_columns = [
        "image_name",
        "mask_name",
        "case_id",
        "merged_mask_name",
        "prediction_name",
    ]

    for column in candidate_columns:
        if column not in metrics_df.columns:
            continue

        stems = metrics_df[column].astype(str).apply(lambda x: Path(x).stem)

        exact_matches = metrics_df[stems == root_name]

        if len(exact_matches) > 0:
            return exact_matches.iloc[0]

    for column in candidate_columns:
        if column not in metrics_df.columns:
            continue

        contains_matches = metrics_df[
            metrics_df[column].astype(str).str.contains(root_name, case=False, regex=False)
        ]

        if len(contains_matches) > 0:
            return contains_matches.iloc[0]

    return None


def get_metric_value(row: pd.Series, preferred_metric: str = "dice"):
    lower_to_original = {
        str(column).lower(): column
        for column in row.index
    }

    if preferred_metric.lower() in lower_to_original:
        value = row[lower_to_original[preferred_metric.lower()]]

        try:
            return float(value)
        except Exception:
            return None

    for key in ["dice_score", "dice_coefficient", "mean_dice"]:
        if key in lower_to_original:
            value = row[lower_to_original[key]]

            try:
                return float(value)
            except Exception:
                return None

    return None


def format_dice(dice_value):
    if dice_value is None or pd.isna(dice_value):
        return "Dice: n/a"

    return f"Dice: {dice_value:.3f}"


def find_prediction_mask_path(
    dataset_key: str,
    split: str,
    folder_name: str,
    training_state: str,
    prompt_mode: str,
    row: pd.Series,
):
    result_split_dir = (
        RESULTS_ROOT
        / dataset_key
        / folder_name
        / training_state
        / prompt_mode
        / split
    )

    merged_dir = result_split_dir / "merged_masks"

    candidates = []

    if "merged_mask_name" in row.index:
        candidates.append(merged_dir / str(row["merged_mask_name"]))

    if "prediction_name" in row.index:
        candidates.append(merged_dir / str(row["prediction_name"]))

    if "case_id" in row.index:
        case_id = str(row["case_id"])
        candidates.append(merged_dir / f"{case_id}.png")
        candidates.append(merged_dir / f"{case_id}_mask.png")
        candidates.append(merged_dir / f"{case_id}_merged.png")

    if "image_name" in row.index:
        image_stem = Path(str(row["image_name"])).stem
        candidates.append(merged_dir / f"{image_stem}.png")
        candidates.append(merged_dir / f"{image_stem}_mask.png")
        candidates.append(merged_dir / f"{image_stem}_merged.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if "case_id" in row.index:
        case_id = str(row["case_id"])
        matches = sorted(merged_dir.glob(f"{case_id}*.png"))

        if matches:
            return matches[0]

    if "image_name" in row.index:
        image_stem = Path(str(row["image_name"])).stem
        matches = sorted(merged_dir.glob(f"{image_stem}*.png"))

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find prediction mask in {merged_dir} for row: {row.to_dict()}"
    )


def load_model_prediction_for_example(
    dataset_key: str,
    split: str,
    root_name: str,
    model_spec: dict,
):
    metrics_path, prompt_mode, folder_name = find_metrics_file_for_model(
        dataset_key=dataset_key,
        split=split,
        model_spec=model_spec,
    )

    if metrics_path is None:
        print(
            f"WARNING: Missing metrics file for "
            f"{dataset_key} | {split} | {model_spec['display_name']}"
        )
        return None

    metrics_df = pd.read_csv(metrics_path)

    row = find_row_for_root(metrics_df, root_name)

    if row is None:
        print(
            f"WARNING: Could not find row for root '{root_name}' in "
            f"{metrics_path} for {model_spec['display_name']}"
        )
        return None

    dice_value = get_metric_value(row, preferred_metric="dice")

    prediction_mask_path = find_prediction_mask_path(
        dataset_key=dataset_key,
        split=split,
        folder_name=folder_name,
        training_state=model_spec["training_state"],
        prompt_mode=prompt_mode,
        row=row,
    )

    pred_mask = read_binary_mask(prediction_mask_path)

    return {
        "display_name": model_spec["display_name"],
        "prompt_mode": prompt_mode,
        "folder_name": folder_name,
        "dice": dice_value,
        "prediction_mask_path": prediction_mask_path,
        "pred_mask": pred_mask,
        "metrics_path": metrics_path,
    }


def plot_dataset_example(dataset_key: str, split: str, root_name: str):
    print("\n" + "=" * 100)
    print(f"Creating qualitative comparison: {dataset_key} | {split} | {root_name}")
    print("=" * 100)

    source_image_path, source_mask_path = find_source_image_and_mask(
        dataset_key=dataset_key,
        split=split,
        root_name=root_name,
    )

    image_np = read_rgb_image(source_image_path)
    gt_mask = read_binary_mask(source_mask_path)

    panels = []

    panels.append(
        {
            "title": "Input Image",
            "bottom": "",
            "image": image_np,
            "mode": "rgb",
        }
    )

    panels.append(
        {
            "title": "GT",
            "bottom": "",
            "image": make_gt_display(gt_mask),
            "mode": "gray",
        }
    )

    loaded_rows = []

    for model_spec in MODEL_ORDER:
        result = load_model_prediction_for_example(
            dataset_key=dataset_key,
            split=split,
            root_name=root_name,
            model_spec=model_spec,
        )

        if result is None:
            blank = np.zeros_like(image_np)
            panels.append(
                {
                    "title": model_spec["display_name"],
                    "bottom": "Dice: n/a",
                    "image": blank,
                    "mode": "rgb",
                }
            )

            loaded_rows.append(
                {
                    "dataset": dataset_key,
                    "split": split,
                    "root_name": root_name,
                    "model": model_spec["display_name"],
                    "status": "missing",
                    "prompt_mode": "",
                    "dice": "",
                    "prediction_mask_path": "",
                    "metrics_path": "",
                }
            )
            continue

        pred_mask = resize_mask_to_image(result["pred_mask"], image_np.shape)

        overlay_np = make_overlay(
            image_np=image_np,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
        )

        panels.append(
            {
                "title": result["display_name"],
                "bottom": format_dice(result["dice"]),
                "image": overlay_np,
                "mode": "rgb",
            }
        )

        loaded_rows.append(
            {
                "dataset": dataset_key,
                "split": split,
                "root_name": root_name,
                "model": result["display_name"],
                "status": "loaded",
                "prompt_mode": result["prompt_mode"],
                "dice": result["dice"],
                "prediction_mask_path": str(result["prediction_mask_path"]),
                "metrics_path": str(result["metrics_path"]),
            }
        )

    n_panels = len(panels)

    fig_width = max(28, PANEL_WIDTH * n_panels)
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

    output_png = OUTPUT_ROOT / f"{dataset_key}_{split}_{root_name}_inference_comparison.png"
    output_csv = OUTPUT_ROOT / f"{dataset_key}_{split}_{root_name}_inference_comparison_loaded_files.csv"

    fig.savefig(output_png, dpi=DPI, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    pd.DataFrame(loaded_rows).to_csv(output_csv, index=False)

    print(f"Saved PNG: {output_png}")
    print(f"Saved CSV: {output_csv}")


def main():
    setup_matplotlib()

    for dataset_key, example_cfg in EXAMPLES.items():
        split = example_cfg["split"]
        root_name = example_cfg["root_name"]

        if root_name.startswith("PUT_"):
            print(
                f"Skipping {dataset_key}. "
                f"Set a real root_name in EXAMPLES first."
            )
            continue

        plot_dataset_example(
            dataset_key=dataset_key,
            split=split,
            root_name=root_name,
        )


if __name__ == "__main__":
    main()
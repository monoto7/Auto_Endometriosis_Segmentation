from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Project path
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Existing project imports
# =============================================================================

from Hybrid.hybrid_utils import (
    find_file_by_root,
    list_image_files,
    read_rgb_image,
    read_binary_mask,
    save_binary_mask,
    save_rgb_image,
    save_grayscale_uint8,
    save_json,
    mask_to_component_boxes,
    compute_binary_metrics,
    make_comparison_overlay,
    Timer,
)

from Hybrid.segformer_inference import (
    build_segformer_model,
    predict_segformer_mask,
)

from Hybrid.surgisam2_fallback_inference import (
    build_surgisam2_predictor,
)

from Hybrid.autobox_posneg_prompt_utils import (
    build_box_posneg_prompt_from_component,
    select_best_candidate_by_segformer_dice,
    apply_dice_fallback,
)


# =============================================================================
# Configuration
# =============================================================================

RESULTS_ROOT = Path(r"F:\Results\SAM_Benchmarking")

STANDARDIZED_DATASETS = {
    "ENID": Path(r"F:\Datasets\Standardized datasets\ENID\ENID 60_20_20 Split"),
    "GLENDA": Path(r"F:\Datasets\Standardized datasets\GLENDA\GLENDA 60_20_20 split"),
    "GLENDA_clean": Path(r"F:\Datasets\Standardized datasets\GLENDA_clean\GLENDA_clean 60_20_20 split"),
}


SEGFORMER_CHECKPOINTS = {
    "ENID": Path(r"F:\Results\SAM_Benchmarking\ENID\SegFormer\trained\checkpoints\best_SegFormer.pt"),
    "GLENDA": Path(r"F:\Results\SAM_Benchmarking\GLENDA\SegFormer\trained\checkpoints\best_SegFormer.pt"),
    "GLENDA_clean": Path(r"F:\Results\SAM_Benchmarking\GLENDA_clean\SegFormer\trained\checkpoints\best_SegFormer.pt"),
}


SEGFORMER_PRETRAINED_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
SEGFORMER_INPUT_SIZE = 512
SEGFORMER_THRESHOLD = 0.5


SURGISAM2_SAM2_REPO_ROOT = Path(r"F:\Models\SAM2")
SURGISAM2_REPO_ROOT = Path(r"F:\Models\SurgiSAM2")
SURGISAM2_CHECKPOINT_PATH = Path(
    r"F:\Models\SurgiSAM2\checkpoints\Curated400_checkpoint_image_predictor.pt"
)
SURGISAM2_MODEL_CFG = "configs/sam2/sam2_hiera_b+.yaml"

SURGISAM2_MULTIMASK_OUTPUT = True
SURGISAM2_USE_BFLOAT16 = True


METHOD_FOLDER_NAME = "SegFormer_SurgiSAM2_AutoBox_PosNeg_Fallback"
TRAINING_STATE_FOLDER = "hybrid"
PROMPT_MODE_FOLDER = "Auto_box_posneg_fallback_dice_0p80"


DATASETS_TO_RUN = [
    "ENID",
    "GLENDA",
    "GLENDA_clean",
]


SPLITS_TO_RUN = [
    "val",
    "test",
]


# Set to 5 for quick debug.
# Set to None for full run.
MAX_IMAGES_DEBUG = None


# SegFormer component extraction settings
BOX_PADDING_PX = 0
BOX_PADDING_RATIO = 0.0
BOX_MIN_COMPONENT_AREA_PX = 0
BOX_MAX_COMPONENTS = None


# New box + positive + negative point prompt settings
NEGATIVE_POINT_FALLBACK_OFFSET_PX = 5


# Fallback rule:
# accept SurgiSAM2 only if Dice(SurgiSAM2 candidate, SegFormer component) > 0.80
# fallback if Dice <= 0.80
DICE_FALLBACK_THRESHOLD = 0.85


SAVE_PROBABILITY_MAPS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Folder helpers
# =============================================================================

def make_output_dirs(dataset_key: str, split_key: str):
    split_output_dir = (
        RESULTS_ROOT
        / dataset_key
        / METHOD_FOLDER_NAME
        / TRAINING_STATE_FOLDER
        / PROMPT_MODE_FOLDER
        / split_key
    )

    dirs = {
        "root": split_output_dir,
        "final_masks": split_output_dir / "merged_masks",
        "final_overlays": split_output_dir / "overlays",
        "segformer_masks": split_output_dir / "segformer_initial_masks",
        "segformer_prompt_masks": split_output_dir / "segformer_prompt_masks_components",
        "segformer_overlays": split_output_dir / "segformer_initial_overlays",
        "probability_maps": split_output_dir / "segformer_probability_maps",
        "prompts": split_output_dir / "auto_prompts",
        "component_decisions": split_output_dir / "component_decisions",
        "surgisam2_selected_components": split_output_dir / "surgisam2_selected_components",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def save_run_config(output_dir: Path):
    config = {
        "method_folder_name": METHOD_FOLDER_NAME,
        "training_state_folder": TRAINING_STATE_FOLDER,
        "prompt_mode_folder": PROMPT_MODE_FOLDER,
        "datasets_to_run": DATASETS_TO_RUN,
        "splits_to_run": SPLITS_TO_RUN,
        "max_images_debug": MAX_IMAGES_DEBUG,
        "segformer_pretrained_model_name": SEGFORMER_PRETRAINED_MODEL_NAME,
        "segformer_input_size": SEGFORMER_INPUT_SIZE,
        "segformer_threshold": SEGFORMER_THRESHOLD,
        "box_padding_px": BOX_PADDING_PX,
        "box_padding_ratio": BOX_PADDING_RATIO,
        "box_min_component_area_px": BOX_MIN_COMPONENT_AREA_PX,
        "box_max_components": BOX_MAX_COMPONENTS,
        "negative_point_fallback_offset_px": NEGATIVE_POINT_FALLBACK_OFFSET_PX,
        "dice_fallback_threshold": DICE_FALLBACK_THRESHOLD,
        "save_probability_maps": SAVE_PROBABILITY_MAPS,
        "device": str(DEVICE),
        "surgisam2_sam2_repo_root": str(SURGISAM2_SAM2_REPO_ROOT),
        "surgisam2_repo_root": str(SURGISAM2_REPO_ROOT),
        "surgisam2_checkpoint_path": str(SURGISAM2_CHECKPOINT_PATH),
        "surgisam2_model_cfg": SURGISAM2_MODEL_CFG,
        "surgisam2_multimask_output": SURGISAM2_MULTIMASK_OUTPUT,
        "surgisam2_use_bfloat16": SURGISAM2_USE_BFLOAT16,
        "fallback_rule": (
            "For each SegFormer connected component, create one tight box, "
            "one positive point inside the component, and four negative points "
            "from background around the component. Run SurgiSAM2 with box + "
            "positive/negative points. Select the candidate with highest Dice "
            "agreement against the SegFormer component. Accept SurgiSAM2 only "
            "if Dice agreement > 0.80. If Dice agreement <= 0.80, fallback to "
            "the original SegFormer component."
        ),
    }

    config_path = output_dir / "run_config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


# =============================================================================
# SurgiSAM2 box + point prediction
# =============================================================================

def predict_surgisam2_candidates_from_box_posneg(
    predictor,
    image_np: np.ndarray,
    box_xyxy,
    point_coords,
    point_labels,
    multimask_output: bool = True,
    use_bfloat16: bool = True,
):
    """
    Runs SurgiSAM2/SAM2 image predictor with:
    - tight box
    - one positive point
    - four negative points

    point_labels:
    - 1 = positive
    - 0 = negative
    """

    box_np = np.asarray(box_xyxy, dtype=np.float32)
    point_coords_np = np.asarray(point_coords, dtype=np.float32)
    point_labels_np = np.asarray(point_labels, dtype=np.int32)

    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got shape {image_np.shape}")

    if DEVICE.type == "cuda" and use_bfloat16:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image_np)
            masks, scores, logits = predictor.predict(
                point_coords=point_coords_np,
                point_labels=point_labels_np,
                box=box_np,
                multimask_output=multimask_output,
            )
    else:
        with torch.inference_mode():
            predictor.set_image(image_np)
            masks, scores, logits = predictor.predict(
                point_coords=point_coords_np,
                point_labels=point_labels_np,
                box=box_np,
                multimask_output=multimask_output,
            )

    masks = np.asarray(masks)
    scores = np.asarray(scores)

    return masks, scores


def save_component_mask(mask_np: np.ndarray, output_path: Path):
    mask_np = (mask_np > 0).astype(np.uint8)
    save_binary_mask(mask_np, output_path)


# =============================================================================
# Main split runner
# =============================================================================

def run_dataset_split(
    dataset_key: str,
    split_key: str,
    segformer_model,
    surgisam2_predictor,
):
    dataset_root = STANDARDIZED_DATASETS[dataset_key]

    image_folder = dataset_root / split_key / "images"
    mask_folder = dataset_root / split_key / "masks"

    if not image_folder.exists():
        raise FileNotFoundError(f"Missing image folder: {image_folder}")

    if not mask_folder.exists():
        raise FileNotFoundError(f"Missing mask folder: {mask_folder}")

    output_dirs = make_output_dirs(
        dataset_key=dataset_key,
        split_key=split_key,
    )

    save_run_config(output_dirs["root"])

    image_files = list_image_files(image_folder)

    if MAX_IMAGES_DEBUG is not None:
        image_files = image_files[:MAX_IMAGES_DEBUG]

    print("\n" + "=" * 100)
    print(f"Running hybrid AutoBox+PosNeg fallback: {dataset_key} | {split_key}")
    print(f"Images: {len(image_files)}")
    print(f"Output: {output_dirs['root']}")
    print(f"Fallback rule: fallback to SegFormer if Dice <= {DICE_FALLBACK_THRESHOLD}")
    print("=" * 100)

    metric_rows = []
    time_rows = []

    for image_index, image_path in enumerate(image_files, start=1):
        root_name = image_path.stem

        mask_path = find_file_by_root(mask_folder, root_name)

        if mask_path is None:
            print(f"WARNING: Missing GT mask for {image_path.name}. Skipping.")
            continue

        print(f"[{image_index}/{len(image_files)}] {root_name}")

        image_np = read_rgb_image(image_path)
        gt_mask = read_binary_mask(mask_path)

        with Timer() as total_timer:
            with Timer() as segformer_timer:
                segformer_mask, segformer_probability = predict_segformer_mask(
                    model=segformer_model,
                    image_np=image_np,
                    device=DEVICE,
                    input_size=SEGFORMER_INPUT_SIZE,
                    threshold=SEGFORMER_THRESHOLD,
                )

            with Timer() as prompt_timer:
                (
                    boxes_xyxy,
                    segformer_prompt_mask,
                    component_infos,
                    segformer_component_masks,
                ) = mask_to_component_boxes(
                    mask_np=segformer_mask,
                    image_shape=image_np.shape,
                    padding_px=BOX_PADDING_PX,
                    padding_ratio=BOX_PADDING_RATIO,
                    min_component_area_px=BOX_MIN_COMPONENT_AREA_PX,
                    max_components=BOX_MAX_COMPONENTS,
                    return_component_masks=True,
                )

                component_prompts = []

                for component_index, box_xyxy in enumerate(boxes_xyxy, start=1):
                    segformer_component_mask = segformer_component_masks[component_index - 1]

                    prompt = build_box_posneg_prompt_from_component(
                        component_mask=segformer_component_mask,
                        fallback_offset_px=NEGATIVE_POINT_FALLBACK_OFFSET_PX,
                    )

                    component_prompts.append(prompt)

            with Timer() as surgisam2_timer:
                final_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

                component_decisions = []
                accepted_count = 0
                fallback_count = 0

                if len(boxes_xyxy) == 0:
                    prompt_status = "empty_segformer_mask"
                else:
                    prompt_status = "multi_box_posneg_fallback_prompt_used"

                    for component_index, box_xyxy in enumerate(boxes_xyxy, start=1):
                        segformer_component_mask = segformer_component_masks[component_index - 1]
                        prompt = component_prompts[component_index - 1]

                        if prompt is None:
                            selected_component_mask = segformer_component_mask.astype(np.uint8)
                            final_mask = np.logical_or(
                                final_mask > 0,
                                selected_component_mask > 0,
                            ).astype(np.uint8)

                            fallback_count += 1

                            component_decisions.append(
                                {
                                    "component_index": component_index,
                                    "box_xyxy": box_xyxy,
                                    "prompt_status": "prompt_building_failed",
                                    "segformer_component_area_px": int((segformer_component_mask > 0).sum()),
                                    "accepted_surgisam2": False,
                                    "fallback_used": True,
                                    "decision": "fallback_to_segformer_prompt_building_failed",
                                    "selected_candidate_index": None,
                                    "selected_surgisam2_score": None,
                                    "best_agreement_dice_with_segformer_component": None,
                                    "best_agreement_iou_with_segformer_component": None,
                                    "dice_fallback_threshold": DICE_FALLBACK_THRESHOLD,
                                    "positive_point_xy": None,
                                    "negative_points_xy": None,
                                    "point_coords_xy": None,
                                    "point_labels": None,
                                    "selected_component_area_px": int((selected_component_mask > 0).sum()),
                                }
                            )

                            continue

                        candidate_masks, candidate_scores = predict_surgisam2_candidates_from_box_posneg(
                            predictor=surgisam2_predictor,
                            image_np=image_np,
                            box_xyxy=prompt["box_xyxy"],
                            point_coords=prompt["point_coords"],
                            point_labels=prompt["point_labels"],
                            multimask_output=SURGISAM2_MULTIMASK_OUTPUT,
                            use_bfloat16=SURGISAM2_USE_BFLOAT16,
                        )

                        selection = select_best_candidate_by_segformer_dice(
                            candidate_masks=candidate_masks,
                            segformer_component_mask=segformer_component_mask,
                        )

                        selected_candidate_index = int(selection["best_index"])
                        selected_surgisam2_score = (
                            float(candidate_scores[selected_candidate_index])
                            if len(candidate_scores) > selected_candidate_index
                            else None
                        )

                        fallback_result = apply_dice_fallback(
                            surgisam_mask=selection["best_mask"],
                            segformer_component_mask=segformer_component_mask,
                            dice_agreement=selection["dice_agreement"],
                            dice_threshold=DICE_FALLBACK_THRESHOLD,
                        )

                        selected_component_mask = fallback_result["final_mask"]

                        final_mask = np.logical_or(
                            final_mask > 0,
                            selected_component_mask > 0,
                        ).astype(np.uint8)

                        if fallback_result["accepted_surgisam2"]:
                            accepted_count += 1
                        else:
                            fallback_count += 1

                        selected_component_path = (
                            output_dirs["surgisam2_selected_components"]
                            / f"{root_name}_component_{component_index:03d}.png"
                        )
                        save_component_mask(
                            selection["best_mask"],
                            selected_component_path,
                        )

                        component_decision = {
                            "component_index": component_index,
                            "box_xyxy": prompt["box_xyxy"],
                            "segformer_component_area_px": int((segformer_component_mask > 0).sum()),
                            "positive_point_xy": prompt["positive_point"],
                            "negative_points_xy": prompt["negative_points"],
                            "point_coords_xy": prompt["point_coords"],
                            "point_labels": prompt["point_labels"],
                            "accepted_surgisam2": bool(fallback_result["accepted_surgisam2"]),
                            "fallback_used": bool(fallback_result["fallback_used"]),
                            "decision": fallback_result["fallback_reason"],
                            "selected_candidate_index": selected_candidate_index,
                            "selected_surgisam2_score": selected_surgisam2_score,
                            "best_agreement_dice_with_segformer_component": float(selection["dice_agreement"]),
                            "best_agreement_iou_with_segformer_component": float(selection["iou_agreement"]),
                            "dice_fallback_threshold": DICE_FALLBACK_THRESHOLD,
                            "selected_component_area_px": int((selected_component_mask > 0).sum()),
                            "raw_surgisam2_selected_area_px": int((selection["best_mask"] > 0).sum()),
                        }

                        component_decisions.append(component_decision)

        final_metrics = compute_binary_metrics(
            pred_mask=final_mask,
            gt_mask=gt_mask,
        )

        segformer_metrics = compute_binary_metrics(
            pred_mask=segformer_mask,
            gt_mask=gt_mask,
        )

        segformer_prompt_metrics = compute_binary_metrics(
            pred_mask=segformer_prompt_mask,
            gt_mask=gt_mask,
        )

        final_overlay = make_comparison_overlay(
            image_np=image_np,
            gt_mask=gt_mask,
            pred_mask=final_mask,
            boxes_xyxy=boxes_xyxy,
            box_color=(180, 0, 255),
            box_width=4,
        )

        segformer_overlay = make_comparison_overlay(
            image_np=image_np,
            gt_mask=gt_mask,
            pred_mask=segformer_mask,
            boxes_xyxy=boxes_xyxy,
            box_color=(180, 0, 255),
            box_width=4,
        )

        final_mask_path = output_dirs["final_masks"] / f"{root_name}.png"
        final_overlay_path = output_dirs["final_overlays"] / f"{root_name}_overlay.png"

        segformer_mask_path = output_dirs["segformer_masks"] / f"{root_name}.png"
        segformer_prompt_mask_path = output_dirs["segformer_prompt_masks"] / f"{root_name}.png"
        segformer_overlay_path = output_dirs["segformer_overlays"] / f"{root_name}_overlay.png"

        prompt_path = output_dirs["prompts"] / f"{root_name}.json"
        component_decision_path = output_dirs["component_decisions"] / f"{root_name}.json"

        save_binary_mask(final_mask, final_mask_path)
        save_rgb_image(final_overlay, final_overlay_path)

        save_binary_mask(segformer_mask, segformer_mask_path)
        save_binary_mask(segformer_prompt_mask, segformer_prompt_mask_path)
        save_rgb_image(segformer_overlay, segformer_overlay_path)

        if SAVE_PROBABILITY_MAPS:
            probability_uint8 = np.clip(
                segformer_probability * 255,
                0,
                255,
            ).astype(np.uint8)

            probability_path = output_dirs["probability_maps"] / f"{root_name}.png"
            save_grayscale_uint8(probability_uint8, probability_path)

        prompt_data = {
            "image_name": image_path.name,
            "root_name": root_name,
            "prompt_type": "multi_tight_box_posneg_fallback",
            "prompt_status": prompt_status,
            "boxes_xyxy": boxes_xyxy,
            "num_boxes": len(boxes_xyxy),
            "component_infos": component_infos,
            "box_padding_px": BOX_PADDING_PX,
            "box_padding_ratio": BOX_PADDING_RATIO,
            "box_min_component_area_px": BOX_MIN_COMPONENT_AREA_PX,
            "box_max_components": BOX_MAX_COMPONENTS,
            "negative_point_fallback_offset_px": NEGATIVE_POINT_FALLBACK_OFFSET_PX,
            "dice_fallback_threshold": DICE_FALLBACK_THRESHOLD,
            "accepted_surgisam2_components": accepted_count,
            "fallback_to_segformer_components": fallback_count,
            "acceptance_rate_components": (
                accepted_count / len(boxes_xyxy)
                if len(boxes_xyxy) > 0
                else 0.0
            ),
            "segformer_pred_area": int((segformer_mask > 0).sum()),
            "segformer_prompt_component_area": int((segformer_prompt_mask > 0).sum()),
            "final_pred_area": int((final_mask > 0).sum()),
            "gt_area": int((gt_mask > 0).sum()),
            "component_decisions": component_decisions,
        }

        save_json(prompt_data, prompt_path)
        save_json(prompt_data, component_decision_path)

        metric_row = {
            "image_name": image_path.name,
            "mask_name": mask_path.name,
            "case_id": root_name,
            "prediction_name": final_mask_path.name,
            "overlay_name": final_overlay_path.name,
            "prompt_status": prompt_status,

            "num_boxes": len(boxes_xyxy),
            "accepted_surgisam2_components": accepted_count,
            "fallback_to_segformer_components": fallback_count,
            "acceptance_rate_components": (
                accepted_count / len(boxes_xyxy)
                if len(boxes_xyxy) > 0
                else 0.0
            ),

            "boxes_xyxy": json.dumps(boxes_xyxy),
            "component_infos": json.dumps(component_infos),
            "component_decisions": json.dumps(component_decisions),

            "negative_point_fallback_offset_px": NEGATIVE_POINT_FALLBACK_OFFSET_PX,
            "dice_fallback_threshold": DICE_FALLBACK_THRESHOLD,

            "dice": final_metrics["dice"],
            "iou": final_metrics["iou"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "tp": final_metrics["tp"],
            "fp": final_metrics["fp"],
            "fn": final_metrics["fn"],
            "tn": final_metrics["tn"],
            "pred_area": final_metrics["pred_area"],
            "gt_area": final_metrics["gt_area"],

            "segformer_initial_dice": segformer_metrics["dice"],
            "segformer_initial_iou": segformer_metrics["iou"],
            "segformer_initial_precision": segformer_metrics["precision"],
            "segformer_initial_recall": segformer_metrics["recall"],
            "segformer_initial_pred_area": segformer_metrics["pred_area"],

            "segformer_prompt_component_dice": segformer_prompt_metrics["dice"],
            "segformer_prompt_component_iou": segformer_prompt_metrics["iou"],
            "segformer_prompt_component_precision": segformer_prompt_metrics["precision"],
            "segformer_prompt_component_recall": segformer_prompt_metrics["recall"],
            "segformer_prompt_component_area": segformer_prompt_metrics["pred_area"],
        }

        time_row = {
            "image_name": image_path.name,
            "case_id": root_name,
            "total_time_sec": total_timer.elapsed,
            "segformer_time_sec": segformer_timer.elapsed,
            "prompt_generation_time_sec": prompt_timer.elapsed,
            "surgisam2_time_sec": surgisam2_timer.elapsed,
            "num_boxes": len(boxes_xyxy),
            "accepted_surgisam2_components": accepted_count,
            "fallback_to_segformer_components": fallback_count,
        }

        metric_rows.append(metric_row)
        time_rows.append(time_row)

    metrics_df = pd.DataFrame(metric_rows)
    times_df = pd.DataFrame(time_rows)

    metrics_path = output_dirs["root"] / "metrics_image_level.csv"
    times_path = output_dirs["root"] / "inference_times.csv"
    summary_path = output_dirs["root"] / "metrics_summary.csv"
    xlsx_path = output_dirs["root"] / "summary.xlsx"

    metrics_df.to_csv(metrics_path, index=False)
    times_df.to_csv(times_path, index=False)

    summary_rows = []

    metric_columns = [
        "dice",
        "iou",
        "precision",
        "recall",
        "segformer_initial_dice",
        "segformer_initial_iou",
        "segformer_initial_precision",
        "segformer_initial_recall",
        "segformer_prompt_component_dice",
        "segformer_prompt_component_iou",
        "segformer_prompt_component_precision",
        "segformer_prompt_component_recall",
        "num_boxes",
        "accepted_surgisam2_components",
        "fallback_to_segformer_components",
        "acceptance_rate_components",
    ]

    for metric_name in metric_columns:
        if metric_name not in metrics_df.columns:
            continue

        values = pd.to_numeric(metrics_df[metric_name], errors="coerce").dropna()

        if len(values) == 0:
            continue

        summary_rows.append(
            {
                "metric": metric_name,
                "n": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
            }
        )

    time_columns = [
        "total_time_sec",
        "segformer_time_sec",
        "prompt_generation_time_sec",
        "surgisam2_time_sec",
    ]

    for time_name in time_columns:
        if time_name not in times_df.columns:
            continue

        values = pd.to_numeric(times_df[time_name], errors="coerce").dropna()

        if len(values) == 0:
            continue

        summary_rows.append(
            {
                "metric": time_name,
                "n": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics_image_level", index=False)
        times_df.to_excel(writer, sheet_name="inference_times", index=False)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved times:   {times_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved xlsx:    {xlsx_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Device: {DEVICE}")

    surgisam2_predictor = build_surgisam2_predictor(
        sam2_repo_root=SURGISAM2_SAM2_REPO_ROOT,
        surgisam2_repo_root=SURGISAM2_REPO_ROOT,
        model_cfg=SURGISAM2_MODEL_CFG,
        checkpoint_path=SURGISAM2_CHECKPOINT_PATH,
        device=DEVICE,
    )

    for dataset_key in DATASETS_TO_RUN:
        checkpoint_path = SEGFORMER_CHECKPOINTS[dataset_key]

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing SegFormer checkpoint: {checkpoint_path}")

        print("\n" + "#" * 100)
        print(f"Loading SegFormer for {dataset_key}")
        print(f"Checkpoint: {checkpoint_path}")
        print("#" * 100)

        segformer_model = build_segformer_model(
            checkpoint_path=checkpoint_path,
            device=DEVICE,
            pretrained_model_name=SEGFORMER_PRETRAINED_MODEL_NAME,
            num_labels=1,
        )

        for split_key in SPLITS_TO_RUN:
            run_dataset_split(
                dataset_key=dataset_key,
                split_key=split_key,
                segformer_model=segformer_model,
                surgisam2_predictor=surgisam2_predictor,
            )

        del segformer_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDONE.")


if __name__ == "__main__":
    main()
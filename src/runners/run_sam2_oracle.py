from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.models.sam2.sam2_wrapper import SAM2FrozenWrapper
from src.utils.mask_utils import (
    load_binary_mask,
    load_mask,
    save_binary_mask,
    merge_binary_masks,
    find_image_path,
    empty_mask_like_image,
    ensure_same_size,
)
from src.utils.visualization import save_overlay
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.metrics import compute_multiclass_metrics

def load_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_json_field(value, default):
    if value is None:
        return default

    if isinstance(value, float) and np.isnan(value):
        return default

    if isinstance(value, list):
        return value

    try:
        return json.loads(value)
    except Exception:
        return default


def build_prompt_from_row(row, prompt_mode: str):
    box = None
    point_coords = None
    point_labels = None

    if prompt_mode == "GT_point":
        point_coords = [[float(row["positive_point_x"]), float(row["positive_point_y"])]]
        point_labels = [1]

    elif prompt_mode == "GT_box":
        box = [
            float(row["bbox_x1"]),
            float(row["bbox_y1"]),
            float(row["bbox_x2"]),
            float(row["bbox_y2"]),
        ]

    elif prompt_mode == "GT_box_point":
        box = [
            float(row["bbox_x1"]),
            float(row["bbox_y1"]),
            float(row["bbox_x2"]),
            float(row["bbox_y2"]),
        ]
        point_coords = [[float(row["positive_point_x"]), float(row["positive_point_y"])]]
        point_labels = [1]

    elif prompt_mode == "GT_box_posneg":
        box = [
            float(row["bbox_x1"]),
            float(row["bbox_y1"]),
            float(row["bbox_x2"]),
            float(row["bbox_y2"]),
        ]
        point_coords = parse_json_field(row.get("point_coords_xy"), [])
        point_labels = parse_json_field(row.get("point_labels"), [])

        if len(point_coords) == 0 or len(point_labels) == 0:
            point_coords = [[float(row["positive_point_x"]), float(row["positive_point_y"])]]
            point_labels = [1]

    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    return box, point_coords, point_labels


def run_one_prompt_mode(
    dataset_name: str,
    dataset_root: Path,
    split_name: str,
    split_cfg: dict,
    output_root: Path,
    prompt_mode: str,
    model: SAM2FrozenWrapper,
    save_cfg: dict,
):
    images_dir = dataset_root / split_cfg["images"]
    masks_dir = dataset_root / split_cfg["masks"]
    prompt_csv = dataset_root / split_cfg["prompts"]

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    if not prompt_csv.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {prompt_csv}")

    prompts_df = pd.read_csv(prompt_csv)

    out_dir = output_root / prompt_mode / split_name
    instance_dir = out_dir / "instance_masks"
    merged_dir = out_dir / "merged_masks"
    overlay_dir = out_dir / "overlays"

    instance_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    inference_rows = []
    metric_rows = []

    grouped = prompts_df.groupby("image_name", sort=True)

    for image_name, image_rows in tqdm(
        grouped,
        desc=f"{dataset_name} | {split_name} | {prompt_mode}",
    ):
        image_path = find_image_path(images_dir, image_name)
        gt_mask_path = masks_dir / f"{Path(image_name).stem}.png"

        if not gt_mask_path.exists():
            raise FileNotFoundError(f"GT mask not found: {gt_mask_path}")

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        gt_mask = load_mask(gt_mask_path)

        model.set_image(image_np)

        lesion_pred_masks = []

        #Create dictionary for storing class predictions
        class_merged_pred = {}



        for _, row in image_rows.iterrows():
            lesion_id = int(row["lesion_id"])
            box, point_coords, point_labels = build_prompt_from_row(row, prompt_mode)

            #Get class ID/grayscaleVal if available
            class_id = int(row["class_id"]) if "class_id" in row else None
            
            start_time = time.perf_counter()

            pred_mask, sam_score, selected_index = model.predict(
                box=box,
                point_coords=point_coords,
                point_labels=point_labels,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            pred_mask = ensure_same_size(pred_mask, gt_mask)
            lesion_pred_masks.append(pred_mask)
            #If class seperation exists, merge output
            if(class_id is not None):
                if(class_id not in class_merged_pred):
                        class_merged_pred[class_id] = np.zeros(gt_mask.shape, dtype=np.uint8)
                class_merged_pred[class_id] = np.maximum(class_merged_pred[class_id], pred_mask)

                instance_name = f"{Path(image_name).stem}_class_{class_id:03d}_lesion_{lesion_id:03d}.png"
            else:
                instance_name = f"{Path(image_name).stem}_lesion_{lesion_id:03d}.png"
            instance_path = instance_dir / instance_name

            if save_cfg.get("instance_masks", True):
                save_binary_mask(pred_mask, instance_path)

            inference_rows.append({
                "dataset": dataset_name,
                "split": split_name,
                "method": "SAM2_frozen",
                "prompt_mode": prompt_mode,
                "image_name": image_name,
                "gt_mask_name": gt_mask_path.name,
                "class_id" : class_id,
                "lesion_id": lesion_id,
                "bbox_xyxy": json.dumps([
                    int(row["bbox_x1"]),
                    int(row["bbox_y1"]),
                    int(row["bbox_x2"]),
                    int(row["bbox_y2"]),
                ]),
                "point_coords_xy": json.dumps(point_coords if point_coords is not None else []),
                "point_labels": json.dumps(point_labels if point_labels is not None else []),
                "sam_score": sam_score,
                "selected_mask_index": selected_index,
                "instance_mask_name": instance_name,
                "inference_time_ms": elapsed_ms,
            })

        if len(lesion_pred_masks) > 0:
            merged_mask = merge_binary_masks(lesion_pred_masks)
        else:
            merged_mask = empty_mask_like_image(image_path)
        

        merged_mask = ensure_same_size(merged_mask, gt_mask)

        merged_name = f"{Path(image_name).stem}.png"
        merged_path = merged_dir / merged_name

        if save_cfg.get("merged_masks", True):
            save_binary_mask(merged_mask, merged_path)

        #Handling for seperate classes to save them as seperate masks
        if save_cfg.get("class_merged_masks", True):
            for key in class_merged_pred:
                class_merged_name =  f"classes/{Path(image_name).stem}_{key:03d}.png"
                class_merged_path = merged_dir / class_merged_name
                save_binary_mask(class_merged_pred[key], class_merged_path)
        
                    
        
        if save_cfg.get("overlays", True):
            overlay_path = overlay_dir / f"{Path(image_name).stem}_overlay.jpg"
            save_overlay(
                image_path=image_path,
                gt_mask=gt_mask,
                pred_mask=merged_mask,
                output_path=overlay_path,
            )


        if len(class_merged_pred) == 0:
            metrics = compute_binary_metrics(merged_mask, gt_mask)

            metrics.update({
            "dataset": dataset_name,
            "split": split_name,
            "method": "SAM2_frozen",
            "prompt_mode": prompt_mode,
            "image_name": image_name,
            "gt_mask_name": gt_mask_path.name,
            "merged_mask_name": merged_name,
            "num_prompt_instances": int(len(image_rows)),
            })
        else:
            for key in class_merged_pred:
                #compute_multiclass_metrics uses the class_id as the grayscale value to compare with in GT.
                metrics = compute_multiclass_metrics(
                    pred_mask=merged_mask,
                    gt_mask=gt_mask,
                    class_g_value=key
                )


                metrics.update({
                    "dataset": dataset_name,
                    "split": split_name,
                    "method": "SAM2_frozen",
                    "prompt_mode": prompt_mode,
                    "image_name": image_name,
                    "gt_mask_name": gt_mask_path.name,
                    "merged_mask_name": merged_name,
                    "class_id": key,
                    "num_prompt_instances": int(len(image_rows)),
                })

        metric_rows.append(metrics)

    inference_df = pd.DataFrame(inference_rows)
    metrics_df = pd.DataFrame(metric_rows)

    inference_csv = out_dir / "inference_results.csv"
    metrics_csv = out_dir / "metrics_image_level.csv"
    summary_csv = out_dir / "metrics_summary.csv"

    inference_df.to_csv(inference_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)

    numeric_cols = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "false_positive_area_px",
        "false_negative_area_px",
        "gt_area_px",
        "pred_area_px",
    ]

    summary = {}
    for col in numeric_cols:
        if col in metrics_df.columns:
            summary[f"{col}_mean"] = metrics_df[col].mean()
            summary[f"{col}_std"] = metrics_df[col].std()
            summary[f"{col}_median"] = metrics_df[col].median()

    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print("\nSaved:")
    print(f"  {inference_csv}")
    print(f"  {metrics_csv}")
    print(f"  {summary_csv}")


def run_experiment(experiment_config_path: str):
    import torch

    experiment_config_path = Path(experiment_config_path)

    exp_cfg = load_yaml(experiment_config_path)
    dataset_cfg = load_yaml(Path(exp_cfg["dataset_config"]))
    model_cfg = load_yaml(Path(exp_cfg["model_config"]))

    dataset_name = dataset_cfg["dataset_name"]
    dataset_root = Path(dataset_cfg["dataset_root"])
    output_root = Path(exp_cfg["output_root"])

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Experiment: {exp_cfg['experiment_name']}")
    print(f"Dataset:    {dataset_name}")
    print(f"Root:       {dataset_root}")
    print(f"Output:     {output_root}")
    print("=" * 80)

    model = SAM2FrozenWrapper(
        sam2_repo_root=model_cfg["sam2_repo_root"],
        checkpoint=model_cfg["checkpoint"],
        model_cfg=model_cfg["model_cfg"],
        device=model_cfg.get("device", "cuda"),
        multimask_output=bool(model_cfg.get("multimask_output", True)),
        use_bfloat16=bool(model_cfg.get("use_bfloat16", True)),
    )

    save_cfg = exp_cfg.get("save", {})

    with torch.inference_mode():
        with model.inference_context():
            for split_name in exp_cfg["splits"]:
                split_cfg = dataset_cfg["splits"][split_name]

                for prompt_mode in exp_cfg["prompt_modes"]:
                    run_one_prompt_mode(
                        dataset_name=dataset_name,
                        dataset_root=dataset_root,
                        split_name=split_name,
                        split_cfg=split_cfg,
                        output_root=output_root,
                        prompt_mode=prompt_mode,
                        model=model,
                        save_cfg=save_cfg,
                    )
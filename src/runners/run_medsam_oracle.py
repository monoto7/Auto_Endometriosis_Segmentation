from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.models.medsam.medsam_wrapper import MedSAMFrozenWrapper
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.metrics import compute_multiclass_metrics
from src.utils.visualization import save_overlay

from src.utils.misc_utils import load_yaml


def load_rgb_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

#No longer converting mask into binary, as we want to handle multiple classes and the data preparation should already do this for single class situations
def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    return mask_np


def save_binary_mask(mask: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(output_path)


def find_image_path(images_dir: Path, image_name: str) -> Path:
    image_path = images_dir / image_name

    if image_path.exists():
        return image_path

    stem = Path(image_name).stem

    for extension in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        candidate = images_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found for {image_name} in {images_dir}")


def find_mask_path(masks_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem

    for extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        candidate = masks_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Mask not found for {image_name} in {masks_dir}")


def parse_box_from_row(row):
    return [
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    ]


def run_one_split(
    dataset_name: str,
    dataset_root: Path,
    split_name: str,
    split_cfg: dict,
    output_root: Path,
    model: MedSAMFrozenWrapper,
    save_cfg: dict,
):
    prompt_mode = "GT_box"

    images_dir = dataset_root / split_cfg["images"]
    masks_dir = dataset_root / split_cfg["masks"]
    prompts_csv = dataset_root / split_cfg["prompts"]

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    if not prompts_csv.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {prompts_csv}")

    prompts_df = pd.read_csv(prompts_csv)

    required_columns = [
        "image_name",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]

    for column in required_columns:
        if column not in prompts_df.columns:
            raise ValueError(f"Required column missing from prompt CSV: {column}")

    out_dir = output_root / prompt_mode / split_name

    instance_dir = out_dir / "instance_masks"
    merged_dir = out_dir / "merged_masks"
    overlay_dir = out_dir / "overlays"

    instance_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    inference_rows = []
    metric_rows = []

    grouped = prompts_df.groupby("image_name", sort=False)

    print("\n" + "=" * 100)
    print(f"Running {dataset_name} | {split_name} | MedSAM | {prompt_mode}")
    print("=" * 100)
    print(f"Images:  {images_dir}")
    print(f"Masks:   {masks_dir}")
    print(f"Prompts: {prompts_csv}")
    print(f"Output:  {out_dir}")
    print(f"Images to process: {len(grouped)}")

    for image_index, (image_name, image_prompts) in enumerate(grouped, start=1):
        print(f"[{image_index}/{len(grouped)}] {image_name}")

        image_path = find_image_path(images_dir, image_name)
        mask_path = find_mask_path(masks_dir, image_name)

        image_rgb = load_rgb_image(image_path)
        gt_mask = load_mask(mask_path)

        model.set_image(image_rgb)

        merged_pred = np.zeros(gt_mask.shape, dtype=np.uint8)
        #Create dictionary for storing class predictions
        class_merged_pred = {}

        for _, row in image_prompts.iterrows():
            lesion_id = int(row["lesion_id"]) if "lesion_id" in row else 0
            #Get class ID/grayscaleVal if available
            class_id = int(row["class_id"]) if "class_id" in row else None
            box_xyxy = parse_box_from_row(row)

            start_time = time.perf_counter()
            pred_mask, mean_probability = model.predict(box_xyxy)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask,
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            merged_pred = np.maximum(merged_pred, pred_mask)

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

            inference_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "model_name": "MedSAM",
                    "training_state": "frozen",
                    "prompt_mode": prompt_mode,
                    "image_name": image_name,
                    "mask_name": Path(mask_path).name,
                    "lesion_id": lesion_id,
                    "class_id": class_id,
                    "bbox_x1": box_xyxy[0],
                    "bbox_y1": box_xyxy[1],
                    "bbox_x2": box_xyxy[2],
                    "bbox_y2": box_xyxy[3],
                    "bbox_xyxy": json.dumps(box_xyxy),
                    "medsam_mean_probability": mean_probability,
                    "inference_time_ms": elapsed_ms,
                    "instance_mask_name": instance_name,
                }
            )

        merged_name = f"{Path(image_name).stem}.png"
        merged_path = merged_dir / merged_name

        if save_cfg.get("merged_masks", True):
            save_binary_mask(merged_pred, merged_path)

        #Handling for seperate classes to save them as seperate masks
        if save_cfg.get("class_merged_masks", True):
            for key in class_merged_pred:
                class_merged_name =  f"classes/{Path(image_name).stem}_{key:03d}.png"
                class_merged_path = merged_dir / class_merged_name
                save_binary_mask(class_merged_pred[key], class_merged_path)


        if save_cfg.get("overlays", True):
            overlay_path = overlay_dir / f"{Path(image_name).stem}_overlay.png"

            save_overlay(
                image_path=image_path,
                gt_mask=gt_mask,
                pred_mask=merged_pred,
                output_path=overlay_path,
            )

        #if to handle whether we generate per class or whole merged metrics
        
        if len(class_merged_pred) == 0:
            metrics = compute_binary_metrics(
                pred_mask=merged_pred,
                gt_mask=gt_mask,
            )


            metric_row = {
                "dataset": dataset_name,
                "split": split_name,
                "model_name": "MedSAM",
                "training_state": "frozen",
                "prompt_mode": prompt_mode,
                "image_name": image_name,
                "mask_name": Path(mask_path).name,
                "num_prompt_instances": int(len(image_prompts)),
            }

            metric_row.update(metrics)
            metric_rows.append(metric_row)
        else:
            for key in class_merged_pred:
                #compute_multiclass_metrics uses the class_id as the grayscale value to compare with in GT.
                metrics = compute_multiclass_metrics(
                    pred_mask=merged_pred,
                    gt_mask=gt_mask,
                    class_g_value=key
                )


                metric_row = {
                    "dataset": dataset_name,
                    "split": split_name,
                    "model_name": "MedSAM",
                    "training_state": "frozen",
                    "prompt_mode": prompt_mode,
                    "image_name": image_name,
                    "mask_name": Path(mask_path).name,
                    "class_id": key,
                    "num_prompt_instances": int(len(image_prompts)),
                }

                metric_row.update(metrics)
                metric_rows.append(metric_row)

        

    inference_df = pd.DataFrame(inference_rows)
    metrics_df = pd.DataFrame(metric_rows)

    inference_csv = out_dir / "inference_results.csv"
    metrics_csv = out_dir / "metrics_image_level.csv"
    summary_csv = out_dir / "metrics_summary.csv"

    inference_df.to_csv(inference_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)

    numeric_cols = metrics_df.select_dtypes(include="number").columns

    summary_df = metrics_df[numeric_cols].agg(
        ["mean", "std", "median", "min", "max"]
    ).T

    summary_df.to_csv(summary_csv)

    print(f"Saved inference CSV: {inference_csv}")
    print(f"Saved image-level metrics CSV: {metrics_csv}")
    print(f"Saved summary CSV: {summary_csv}")


def run_experiment(experiment_config_path):
    experiment_config_path = Path(experiment_config_path)
    project_root = Path(__file__).resolve().parents[2]

    exp_cfg = load_yaml(experiment_config_path)

    dataset_cfg_path = project_root / exp_cfg["dataset_config"]
    model_cfg_path = project_root / exp_cfg["model_config"]

    dataset_cfg = load_yaml(dataset_cfg_path)
    model_cfg = load_yaml(model_cfg_path)

    dataset_name = dataset_cfg["dataset_name"]
    dataset_root = Path(dataset_cfg["dataset_root"])
    output_root = Path(exp_cfg["output_root"])

    prompt_modes = exp_cfg.get("prompt_modes", ["GT_box"])

    if prompt_modes != ["GT_box"]:
        raise ValueError("MedSAM runner currently supports only prompt_modes: ['GT_box']")

    model = MedSAMFrozenWrapper(
        medsam_repo_root=model_cfg["medsam_repo_root"],
        checkpoint=model_cfg["checkpoint"],
        device=model_cfg.get("device", "cuda:0"),
        image_size=model_cfg.get("image_size", 1024),
    )

    save_cfg = exp_cfg.get("save", {})

    for split_name in exp_cfg["splits"]:
        split_cfg = dataset_cfg["splits"][split_name]

        run_one_split(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            split_name=split_name,
            split_cfg=split_cfg,
            output_root=output_root,
            model=model,
            save_cfg=save_cfg,
        )
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from src.models.yolo_seg.yolo_seg_model import build_yolo_seg_model
from src.evaluation.metrics import compute_binary_metrics
from src.utils.visualization import save_overlay

#Utils for supporting multiclass case
from src.utils.mask_utils import load_mask
from src.utils.mask_utils import save_mask

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]


def load_yaml(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def find_original_image_path(images_dir: Path, image_name: str) -> Path:
    direct = images_dir / image_name

    if direct.exists():
        return direct

    stem = Path(image_name).stem

    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found: {image_name} in {images_dir}")


def find_original_mask_path(masks_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem

    for extension in IMAGE_EXTENSIONS:
        candidate = masks_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Mask not found for image: {image_name} in {masks_dir}")


def load_binary_mask(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    return (mask_np > 0).astype(np.uint8) * 255


def list_split_images(images_dir: Path):
    image_paths = []

    for extension in IMAGE_EXTENSIONS:
        image_paths.extend(sorted(images_dir.glob(f"*{extension}")))

    return sorted(image_paths)


def save_dataframe_csv_xlsx(df: pd.DataFrame, csv_path: Path, xlsx_path: Path, sheet_name: str):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter

            for cell in column_cells:
                value_length = len(str(cell.value)) if cell.value is not None else 0
                max_length = max(max_length, value_length)

            worksheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 10),
                50,
            )


def copy_ultralytics_training_history(output_root: Path):
    source_csv = output_root / "ultralytics" / "train" / "results.csv"

    if not source_csv.exists():
        return

    history_df = pd.read_csv(source_csv)

    target_csv = output_root / "training_history.csv"
    target_xlsx = output_root / "training_history.xlsx"

    save_dataframe_csv_xlsx(
        df=history_df,
        csv_path=target_csv,
        xlsx_path=target_xlsx,
        sheet_name="training_history",
    )

    print(f"Saved training history CSV:  {target_csv}")
    print(f"Saved training history XLSX: {target_xlsx}")


def save_training_time_summary(
    output_root: Path,
    dataset_name: str,
    model_name: str,
    train_seconds: float,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
):
    summary_df = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "model_name": model_name,
                "training_state": "trained",
                "total_training_time_seconds": train_seconds,
                "total_training_time_minutes": train_seconds / 60.0,
                "best_checkpoint_path": str(best_checkpoint_path),
                "last_checkpoint_path": str(last_checkpoint_path),
            }
        ]
    )

    csv_path = output_root / "training_time_summary.csv"
    xlsx_path = output_root / "training_time_summary.xlsx"

    save_dataframe_csv_xlsx(
        df=summary_df,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        sheet_name="training_time_summary",
    )

    print(f"Saved training time CSV:  {csv_path}")
    print(f"Saved training time XLSX: {xlsx_path}")


def result_to_merged_mask_and_prompt_rows(
    result,
    image_name: str,
    gt_shape,
    conf_threshold: float,
    classes:list
):
    height, width = gt_shape

    merged_mask = np.zeros((height, width), dtype=np.uint8)
    prompt_rows = []

    if result.masks is None or result.boxes is None:
        return merged_mask, prompt_rows

    masks_tensor = result.masks.data
    boxes_tensor = result.boxes.xyxy
    conf_tensor = result.boxes.conf
    cls_tensor = result.boxes.cls

    if masks_tensor is None or boxes_tensor is None:
        return merged_mask, prompt_rows

    masks_np = masks_tensor.detach().cpu().numpy()
    boxes_np = boxes_tensor.detach().cpu().numpy()
    conf_np = conf_tensor.detach().cpu().numpy()
    cls_np = cls_tensor.detach().cpu().numpy()

    for instance_index in range(masks_np.shape[0]):
        confidence = float(conf_np[instance_index])

        if confidence < conf_threshold:
            continue
        instance_mask = (masks_np[instance_index] > 0.5).astype(np.uint8) * classes[int(cls_np[instance_index])]

        if instance_mask.shape != (height, width):
            instance_mask = cv2.resize(
                instance_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        merged_mask = np.maximum(merged_mask, instance_mask)

        x1, y1, x2, y2 = boxes_np[instance_index]

        x1 = int(round(max(0, min(width - 1, x1))))
        y1 = int(round(max(0, min(height - 1, y1))))
        x2 = int(round(max(0, min(width - 1, x2))))
        y2 = int(round(max(0, min(height - 1, y2))))

        if x2 <= x1 or y2 <= y1:
            continue

        bbox_xyxy = [x1, y1, x2, y2]

        prompt_rows.append(
            {
                "image_name": image_name,
                "instance_index": instance_index,
                "source_model": "YOLO11s_seg",
                "class_id": int(cls_np[instance_index]),
                "class_name": "lesion",
                "confidence": confidence,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_xyxy": str(bbox_xyxy),
                "prompt_mode": "Auto_YOLO_box",
            }
        )

    return merged_mask, prompt_rows


def evaluate_yolo_split(
    model,
    dataset_name: str,
    model_name: str,
    dataset_root: Path,
    split_cfg: dict,
    output_root: Path,
    split_name: str,
    model_cfg: dict,
    save_cfg: dict,
    classes : list = ["255"],
):
    prompt_mode = "No_prompt"

    images_dir = dataset_root / split_cfg["images"]
    masks_dir = dataset_root / split_cfg["masks"]

    out_dir = output_root / prompt_mode / split_name
    merged_dir = out_dir / "merged_masks"
    overlay_dir = out_dir / "overlays"

    merged_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    auto_prompts_dir = output_root / "auto_prompts"
    auto_prompts_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_split_images(images_dir)

    imgsz = int(model_cfg.get("imgsz", 640))
    conf = float(model_cfg.get("conf", 0.25))
    iou = float(model_cfg.get("iou", 0.50))
    max_det = int(model_cfg.get("max_det", 100))
    retina_masks = bool(model_cfg.get("retina_masks", True))
    device = model_cfg.get("device", 0)

    inference_rows = []
    metric_rows = []
    all_prompt_rows = []

    total_inference_time_ms = 0.0

    for image_path in image_paths:
        image_name = image_path.name
        mask_path = find_original_mask_path(masks_dir, image_name)
        gt_mask = load_mask(mask_path)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            retina_masks=retina_masks,
            device=device,
            verbose=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        total_inference_time_ms += elapsed_ms

        result = results[0]

        pred_mask, prompt_rows = result_to_merged_mask_and_prompt_rows(
            result=result,
            image_name=image_name,
            gt_shape=gt_mask.shape,
            conf_threshold=conf,
            classes=classes
        )

        all_prompt_rows.extend(prompt_rows)

        merged_name = f"{image_path.stem}.png"
        merged_path = merged_dir / merged_name

        if save_cfg.get("merged_masks", True):
            save_mask(pred_mask, merged_path)

        if save_cfg.get("overlays", True):
            overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"

            save_overlay(
                image_path=image_path,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                output_path=overlay_path,
            )

        metrics = compute_binary_metrics(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
        )

        inference_rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "model_name": model_name,
                "training_state": "trained",
                "prompt_mode": prompt_mode,
                "image_name": image_name,
                "mask_name": mask_path.name,
                "confidence_threshold": conf,
                "iou_threshold": iou,
                "max_det": max_det,
                "num_pred_instances": len(prompt_rows),
                "inference_time_ms": elapsed_ms,
                "merged_mask_name": merged_name,
            }
        )

        metric_row = {
            "dataset": dataset_name,
            "split": split_name,
            "model_name": model_name,
            "training_state": "trained",
            "prompt_mode": prompt_mode,
            "image_name": image_name,
            "mask_name": mask_path.name,
            "num_prompt_instances": len(prompt_rows),
        }

        metric_row.update(metrics)
        metric_rows.append(metric_row)

    inference_df = pd.DataFrame(inference_rows)
    metrics_df = pd.DataFrame(metric_rows)
    prompts_df = pd.DataFrame(all_prompt_rows)

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

    if save_cfg.get("auto_box_prompts", True):
        prompts_csv = auto_prompts_dir / f"{model_name}_{split_name}_auto_box_prompts.csv"
        prompts_df.to_csv(prompts_csv, index=False)
        print(f"Saved auto box prompts: {prompts_csv}")

    time_summary_df = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "split": split_name,
                "model_name": model_name,
                "training_state": "trained",
                "prompt_mode": prompt_mode,
                "num_images": len(image_paths),
                "total_inference_time_seconds": total_inference_time_ms / 1000.0,
                "mean_inference_time_ms": float(inference_df["inference_time_ms"].mean()),
                "std_inference_time_ms": float(inference_df["inference_time_ms"].std()),
                "median_inference_time_ms": float(inference_df["inference_time_ms"].median()),
                "min_inference_time_ms": float(inference_df["inference_time_ms"].min()),
                "max_inference_time_ms": float(inference_df["inference_time_ms"].max()),
            }
        ]
    )

    time_summary_csv = out_dir / "inference_time_summary.csv"
    time_summary_df.to_csv(time_summary_csv, index=False)

    print(f"Saved inference CSV: {inference_csv}")
    print(f"Saved metrics CSV:   {metrics_csv}")
    print(f"Saved summary CSV:   {summary_csv}")
    print(f"Saved time summary:  {time_summary_csv}")

    return metrics_df


def train_and_evaluate(experiment_config_path):
    experiment_config_path = Path(experiment_config_path)
    project_root = Path(__file__).resolve().parents[2]

    exp_cfg = load_yaml(experiment_config_path)

    dataset_cfg = load_yaml(project_root / exp_cfg["dataset_config"])
    model_cfg = load_yaml(project_root / exp_cfg["model_config"])

    dataset_name = dataset_cfg["dataset_name"]
    dataset_root = Path(dataset_cfg["dataset_root"])
    yolo_dataset_yaml = Path(exp_cfg["yolo_dataset_yaml"])
    output_root = Path(exp_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    model_name = model_cfg.get("model_name", "YOLO11s_seg")
    pretrained_model = model_cfg.get("pretrained_model", "yolo11s-seg.pt")

    train_split = exp_cfg["splits"]["train"]
    val_split = exp_cfg["splits"]["val"]
    test_split = exp_cfg["splits"]["test"]

    val_cfg = dataset_cfg["splits"][val_split]
    test_cfg = dataset_cfg["splits"][test_split]

    classes = dataset_cfg["classes"]

    if not yolo_dataset_yaml.exists():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {yolo_dataset_yaml}")

    print("=" * 100)
    print(f"Training {model_name} on {dataset_name}")
    print(f"Dataset root:       {dataset_root}")
    print(f"YOLO dataset YAML:  {yolo_dataset_yaml}")
    print(f"Output root:        {output_root}")
    print(f"Pretrained model:   {pretrained_model}")
    print("=" * 100)

    model = build_yolo_seg_model(pretrained_model)

    train_start = time.perf_counter()

    model.train(
        data=str(yolo_dataset_yaml),
        task="segment",
        project=str(output_root / "ultralytics"),
        name="train",
        exist_ok=True,
        imgsz=int(model_cfg.get("imgsz", 640)),
        epochs=int(model_cfg.get("epochs", 100)),
        batch=int(model_cfg.get("batch", 8)),
        workers=int(model_cfg.get("workers", 4)),
        device=model_cfg.get("device", 0),
        optimizer=model_cfg.get("optimizer", "auto"),
        lr0=float(model_cfg.get("lr0", 0.001)),
        weight_decay=float(model_cfg.get("weight_decay", 0.0005)),
        patience=int(model_cfg.get("patience", 20)),
        save_period=int(model_cfg.get("save_period", -1)),
        plots=bool(model_cfg.get("plots", True)),
    )

    train_seconds = time.perf_counter() - train_start

    best_checkpoint_path = output_root / "ultralytics" / "train" / "weights" / "best.pt"
    last_checkpoint_path = output_root / "ultralytics" / "train" / "weights" / "last.pt"

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint_path}")

    copy_ultralytics_training_history(output_root)

    save_training_time_summary(
        output_root=output_root,
        dataset_name=dataset_name,
        model_name=model_name,
        train_seconds=train_seconds,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )

    print("=" * 100)
    print(f"Loaded best YOLO checkpoint: {best_checkpoint_path}")
    print("=" * 100)

    best_model = build_yolo_seg_model(str(best_checkpoint_path))
    save_cfg = exp_cfg.get("save", {})

    evaluate_yolo_split(
        model=best_model,
        dataset_name=dataset_name,
        model_name=model_name,
        dataset_root=dataset_root,
        split_cfg=val_cfg,
        output_root=output_root,
        split_name="val",
        model_cfg=model_cfg,
        save_cfg=save_cfg,
        classes=classes
    )

    evaluate_yolo_split(
        model=best_model,
        dataset_name=dataset_name,
        model_name=model_name,
        dataset_root=dataset_root,
        split_cfg=test_cfg,
        output_root=output_root,
        split_name="test",
        model_cfg=model_cfg,
        save_cfg=save_cfg,
        classes=classes
    )

    print(f"{model_name} training and evaluation finished for {dataset_name}.")
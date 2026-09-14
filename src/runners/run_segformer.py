from pathlib import Path
import time

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.segmentation_dataset import BinarySegmentationDataset
from src.datasets.segmentation_dataset import GrayscaleSegmentationDataset
from src.models.segformer.segformer_model import build_segformer_model_multiclass
from src.models.segformer.segformer_model import build_segformer_model_binary
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.metrics import compute_multiclass_metrics
from src.utils.visualization import save_overlay



def load_yaml(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def forward_segformer_logits(model, images, target_size):
    outputs = model(pixel_values=images)
    logits = outputs.logits

    logits = torch_functional.interpolate(
        logits,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    return logits


def dice_loss_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    cardinality = torch.sum(probs + targets, dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)

    return 1.0 - dice.mean()

def dice_loss_from_logits_multiclass(logits, targets, eps=1e-7):
    probs = torch.softmax(logits,dim=1)

    #Need to handle each class seperately, so only final 2 are summed over
    dims = (2, 3)

    #Switch targets to be one_hot_encoded in each class
    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

    intersection = torch.sum(probs * one_hot_targets, dims)
    cardinality = torch.sum(probs + one_hot_targets, dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)

    return 1.0 - dice.mean()

def combined_loss(logits, targets, dice_weight=0.5, bce_weight=0.5):
    dice = dice_loss_from_logits(logits, targets)
    bce = torch_functional.binary_cross_entropy_with_logits(logits, targets)

    return dice_weight * dice + bce_weight * bce

def combined_multiclass_loss(logits, targets, dice_weight=0.5, ce_weight=0.5):
    dice = dice_loss_from_logits_multiclass(logits, targets)
    ce = torch_functional.cross_entropy(logits, targets)

    return dice_weight * dice + ce_weight * ce


def save_binary_mask(mask: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(output_path)


def remove_small_components(binary_mask: np.ndarray, min_area_px: int):
    if min_area_px <= 0:
        return binary_mask

    mask = (binary_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area_px:
            cleaned[labels == label_id] = 1

    return cleaned.astype(np.uint8) * 255


def probability_to_binary_mask(probability_map, threshold, postprocessing_cfg):
    binary_mask = (probability_map >= threshold).astype(np.uint8) * 255

    if postprocessing_cfg.get("remove_small_components", False):
        min_area_px = int(postprocessing_cfg.get("min_component_area_px", 0))
        binary_mask = remove_small_components(binary_mask, min_area_px)

    return binary_mask

#Used instead so multiclass is supported, should still support binary
def probability_to_mask(probability_map, threshold, postprocessing_cfg, classes):
    mask = np.zeros(probability_map[0].shape)
    for i in range(probability_map.shape[0]-1):
        mask[probability_map[i+1] >= threshold] = int(classes[i])

        #TODO fix below so it works with multiclass
        if postprocessing_cfg.get("remove_small_components", False):
            min_area_px = int(postprocessing_cfg.get("min_component_area_px", 0))
            mask = remove_small_components(mask, min_area_px)

    return mask


def find_original_image_path(images_dir: Path, image_name: str) -> Path:
    direct = images_dir / image_name

    if direct.exists():
        return direct

    stem = Path(image_name).stem

    for extension in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        candidate = images_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found: {image_name} in {images_dir}")


def find_original_mask_path(masks_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem

    for extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        candidate = masks_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Mask not found for image: {image_name} in {masks_dir}")


def load_binary_mask(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    return (mask_np > 0).astype(np.uint8) * 255

#handling for grayscale masks
def load_mask(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    return mask_np


def train_one_epoch(model, dataloader, optimizer, device, loss_cfg):
    model.train()

    losses = []

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = forward_segformer_logits(
            model=model,
            images=images,
            target_size=masks.shape[-2:],
        )

        loss = combined_multiclass_loss(
            logits=logits,
            targets=masks,
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
            ce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        )

        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses))


@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    device,
    loss_cfg,
    threshold,
    postprocessing_cfg,
):
    model.eval()

    losses = []
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = forward_segformer_logits(
            model=model,
            images=images,
            target_size=masks.shape[-2:],
        )

        loss = combined_multiclass_loss(
            logits=logits,
            targets=masks,
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
            ce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        )

        losses.append(float(loss.detach().cpu().item()))

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()

        #TODO rework below, though it's not used for training so maybe can ignore for now
        for i in range(probs.shape[0]):
            pred_mask = probability_to_binary_mask(
                probability_map=probs[i, 0],
                threshold=threshold,
                postprocessing_cfg=postprocessing_cfg,
            )

            gt_mask = (masks_np[i, 0] > 0).astype(np.uint8) * 255

            metrics = compute_binary_metrics(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
            )

            dice_scores.append(metrics["dice"])
            iou_scores.append(metrics["iou"])
            precision_scores.append(metrics["precision"])
            recall_scores.append(metrics["recall"])

    return {
        "val_loss": float(np.mean(losses)),
        "val_dice": float(np.mean(dice_scores)),
        "val_iou": float(np.mean(iou_scores)),
        "val_precision": float(np.mean(precision_scores)),
        "val_recall": float(np.mean(recall_scores)),
    }


def save_training_history(history_rows, output_root: Path):
    history_df = pd.DataFrame(history_rows)

    csv_path = output_root / "training_history.csv"
    xlsx_path = output_root / "training_history.xlsx"

    history_df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        history_df.to_excel(writer, sheet_name="training_history", index=False)

        worksheet = writer.sheets["training_history"]
        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter

            for cell in column_cells:
                value_length = len(str(cell.value)) if cell.value is not None else 0
                max_length = max(max_length, value_length)

            worksheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 10),
                25,
            )

    return history_df


def save_loss_curve(history_df: pd.DataFrame, output_root: Path):
    curves_dir = output_root / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Training loss",
        linewidth=2,
    )

    ax.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation loss",
        linewidth=2,
    )

    ax.set_title("Training and validation loss", fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=12, frameon=False)

    fig.tight_layout()

    output_path = curves_dir / "loss_curve.png"
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


@torch.no_grad()
def collect_probabilities_for_split(
    model,
    dataloader,
    dataset_root: Path,
    split_cfg: dict,
    device,
):
    model.eval()

    images_dir = dataset_root / split_cfg["images"]
    masks_dir = dataset_root / split_cfg["masks"]

    records = []

    for batch in dataloader:
        images = batch["image"].to(device)
        image_names = batch["image_name"]

        logits = forward_segformer_logits(
            model=model,
            images=images,
            target_size=images.shape[-2:],
        )

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        for i, image_name in enumerate(image_names):
            mask_path = find_original_mask_path(masks_dir, image_name)
            #Load mask instead of binary mask since preprocessing already handles binarization where desired
            gt_mask = load_mask(mask_path)


            #Rescale each given probability mask
            #May behave strangely with multiple classes and linear interpolation, unsure.
            original_h, original_w = gt_mask.shape
            rescaled = np.zeros((probs.shape[1],original_h,original_w))
            for y in range(probs.shape[1]):
                rescaled[y] = cv2.resize(
                    probs[i, y],
                    (original_w, original_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            image_path = find_original_image_path(images_dir, image_name)

            records.append(
                {
                    "image_name": image_name,
                    "probability_map": rescaled,
                    "gt_mask": gt_mask,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )

    return records


def run_threshold_sweep(records, thresholds, postprocessing_cfg, classes = ["255"]):
    rows = []

    for threshold in thresholds:
        metric_rows = []

        for record in records:
            pred_mask = probability_to_mask(
                probability_map=record["probability_map"],
                threshold=float(threshold),
                postprocessing_cfg=postprocessing_cfg,
                #classes added to support multiclass
                classes = classes,
            )

            for i in classes:
                metrics = compute_multiclass_metrics(
                    pred_mask= (pred_mask[int(i)]==int(i)).astype(int),
                    gt_mask=record["gt_mask"],
                    class_g_value=int(i)
                )
                metric_rows.append(metrics)
            

            

        metric_df = pd.DataFrame(metric_rows)

        row = {
            "threshold": float(threshold),
            "dice": float(metric_df["dice"].mean()),
            "iou": float(metric_df["iou"].mean()),
            "precision": float(metric_df["precision"].mean()),
            "recall": float(metric_df["recall"].mean()),
        }

        if "specificity" in metric_df.columns:
            row["specificity"] = float(metric_df["specificity"].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def save_threshold_sweep(threshold_df: pd.DataFrame, output_root: Path):
    csv_path = output_root / "threshold_sweep_val.csv"
    xlsx_path = output_root / "threshold_sweep_val.xlsx"

    threshold_df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        threshold_df.to_excel(writer, sheet_name="threshold_sweep_val", index=False)

        worksheet = writer.sheets["threshold_sweep_val"]
        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter

            for cell in column_cells:
                value_length = len(str(cell.value)) if cell.value is not None else 0
                max_length = max(max_length, value_length)

            worksheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 10),
                25,
            )

    curves_dir = output_root / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        threshold_df["threshold"],
        threshold_df["dice"],
        marker="o",
        label="Dice",
        linewidth=2,
    )

    ax.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        marker="o",
        label="Precision",
        linewidth=2,
    )

    ax.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        marker="o",
        label="Recall",
        linewidth=2,
    )

    ax.set_title("Validation threshold sweep", fontsize=16, fontweight="bold")
    ax.set_xlabel("Threshold", fontsize=14)
    ax.set_ylabel("Metric", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=12, frameon=False)

    fig.tight_layout()

    plot_path = curves_dir / "threshold_sweep_val.png"
    fig.savefig(plot_path, dpi=500)
    plt.close(fig)

    print(f"Saved threshold sweep CSV:  {csv_path}")
    print(f"Saved threshold sweep XLSX: {xlsx_path}")
    print(f"Saved threshold sweep plot: {plot_path}")


@torch.no_grad()
def evaluate_records_and_save(
    records,
    dataset_name: str,
    model_name: str,
    output_root: Path,
    split_name: str,
    threshold: float,
    postprocessing_cfg,
    save_cfg,
    inference_time_ms_per_image=None,
    classes = ["255"],
):
    prompt_mode = "No_prompt"

    out_dir = output_root / prompt_mode / split_name
    merged_dir = out_dir / "merged_masks"
    overlay_dir = out_dir / "overlays"

    merged_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    inference_rows = []
    metric_rows = []

    for record in records:
        image_name = record["image_name"]
        image_path = record["image_path"]
        mask_path = record["mask_path"]
        gt_mask = record["gt_mask"]
        #TODO implement value to save masks based upon colourvalue rather than binary
        #colourVal = record["mask"]

        pred_mask = probability_to_mask(
            probability_map=record["probability_map"],
            threshold=threshold,
            postprocessing_cfg=postprocessing_cfg,
            classes = classes
        )

        merged_name = f"{Path(image_name).stem}.png"
        merged_path = merged_dir / merged_name

        if save_cfg.get("merged_masks", True):
            save_binary_mask(pred_mask, merged_path)

        if save_cfg.get("overlays", True):
            overlay_path = overlay_dir / f"{Path(image_name).stem}_overlay.png"

            save_overlay(
                image_path=image_path,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                output_path=overlay_path,
            )
        metrics = None
        if(True or len(classes)==1):
            metrics = compute_binary_metrics(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
            )
        else:
            for class_name in classes:
                #TODO needs rework to properly support below
                metrics = compute_multiclass_metrics(
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                    class_g_value=class_name
                )

        inference_rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "model_name": model_name,
                "training_state": "trained",
                "prompt_mode": prompt_mode,
                "image_name": image_name,
                "mask_name": Path(mask_path).name,
                "threshold": threshold,
                "postprocess_remove_small_components": postprocessing_cfg.get(
                    "remove_small_components",
                    False,
                ),
                "postprocess_min_component_area_px": postprocessing_cfg.get(
                    "min_component_area_px",
                    0,
                ),
                "inference_time_ms": inference_time_ms_per_image,
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
            "mask_name": Path(mask_path).name,
            "num_prompt_instances": 0,
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

    return metrics_df


@torch.no_grad()
def estimate_inference_time_per_image(model, dataloader, device):
    model.eval()

    total_images = 0
    total_time_ms = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        _ = forward_segformer_logits(
            model=model,
            images=images,
            target_size=images.shape[-2:],
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        total_time_ms += elapsed_ms
        total_images += images.shape[0]

    if total_images == 0:
        return None

    return total_time_ms / total_images


def train_and_evaluate(experiment_config_path):
    experiment_config_path = Path(experiment_config_path)
    project_root = Path(__file__).resolve().parents[2]

    exp_cfg = load_yaml(experiment_config_path)

    dataset_cfg = load_yaml(project_root / exp_cfg["dataset_config"])
    model_cfg = load_yaml(project_root / exp_cfg["model_config"])

    dataset_name = dataset_cfg["dataset_name"]
    dataset_root = Path(dataset_cfg["dataset_root"])
    model_name = model_cfg.get("model_name", "SegFormer")
    output_root = Path(exp_cfg["output_root"])

    device = torch.device(
        model_cfg.get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )

    
    image_size = int(model_cfg.get("image_size", 512))
    batch_size = int(model_cfg.get("batch_size", 4))
    num_workers = int(model_cfg.get("num_workers", 4))
    epochs = int(model_cfg.get("epochs", 100))
    threshold = float(model_cfg.get("threshold", 0.5))
    postprocessing_cfg = model_cfg.get("postprocessing", {})

    output_root.mkdir(parents=True, exist_ok=True)

    train_split = exp_cfg["splits"]["train"]
    val_split = exp_cfg["splits"]["val"]
    test_split = exp_cfg["splits"]["test"]

    train_cfg = dataset_cfg["splits"][train_split]
    val_cfg = dataset_cfg["splits"][val_split]
    test_cfg = dataset_cfg["splits"][test_split]

    #Allow experiment config to override model config
    num_labels = model_cfg.get("num_labels", 1)
    num_labels = exp_cfg.get("num_labels", num_labels)

    #Choose dataset loader based upon number of labels to prevent binarization
    Dataset = BinarySegmentationDataset

    if num_labels > 1:
        Dataset = GrayscaleSegmentationDataset
    classes = dataset_cfg.get("classes", ["255"])
    

    train_dataset = Dataset(
        images_dir=dataset_root / train_cfg["images"],
        masks_dir=dataset_root / train_cfg["masks"],
        image_size=image_size,
        augment=True,
        classes = classes
    )

    val_dataset = Dataset(
        images_dir=dataset_root / val_cfg["images"],
        masks_dir=dataset_root / val_cfg["masks"],
        image_size=image_size,
        augment=False,
        classes = classes
    )

    test_dataset = Dataset(
        images_dir=dataset_root / test_cfg["images"],
        masks_dir=dataset_root / test_cfg["masks"],
        image_size=image_size,
        augment=False,
        classes = classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    #Moved num_labels reading earlier to allow for its use in determining other behaviour
    build_segformer_model = build_segformer_model_binary
    if(num_labels>1):
        build_segformer_model = build_segformer_model_multiclass

    model = build_segformer_model(
        pretrained_model_name=model_cfg.get("pretrained_model_name", "nvidia/mit-b2"),
        num_labels=int(num_labels),
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("learning_rate", 6e-5)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-5)),
    )

    scheduler_cfg = model_cfg.get("scheduler", {})
    scheduler = None

    if scheduler_cfg.get("use_reduce_on_plateau", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 8)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )

    early_cfg = model_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 20))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    epochs_without_improvement = 0

    safe_model_name = model_name.replace("/", "_").replace("\\", "_")

    best_val_dice = -1.0
    best_checkpoint_path = output_root / "checkpoints" / f"best_{safe_model_name}.pt"
    last_checkpoint_path = output_root / "checkpoints" / f"last_{safe_model_name}.pt"

    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history_rows = []
    save_cfg = exp_cfg.get("save", {})

    print("=" * 100)
    print(f"Training {model_name} on {dataset_name}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output root:  {output_root}")
    print(f"Device:       {device}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")
    print(f"Test images:  {len(test_dataset)}")
    print(f"Backbone:     {model_cfg.get('pretrained_model_name', 'nvidia/mit-b2')}")
    print(f"Threshold:    {threshold}")
    print(f"Postprocess:  {postprocessing_cfg}")
    print(f"Early stopping enabled: {early_enabled}")
    print(f"Early stopping patience: {early_patience}")
    print(f"Early stopping min_delta: {early_min_delta}")
    print("=" * 100)

    for epoch in range(1, epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_cfg=model_cfg.get("loss", {}),
        )

        val_stats = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_cfg=model_cfg.get("loss", {}),
            threshold=threshold,
            postprocessing_cfg=postprocessing_cfg,
        )

        val_loss = val_stats["val_loss"]
        val_dice = val_stats["val_dice"]

        history_rows.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_stats["val_iou"],
                "val_precision": val_stats["val_precision"],
                "val_recall": val_stats["val_recall"],
            }
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_precision={val_stats['val_precision']:.4f} | "
            f"val_recall={val_stats['val_recall']:.4f}"
        )

        improved = val_dice > best_val_dice + early_min_delta

        if improved:
            best_val_dice = val_dice
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_cfg": model_cfg,
                    "epoch": epoch,
                    "best_val_dice": best_val_dice,
                    "selection_metric": "val_dice",
                    "selection_min_delta": early_min_delta,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(val_dice)

        history_df = save_training_history(history_rows, output_root)
        save_loss_curve(history_df, output_root)

        if early_enabled and epochs_without_improvement >= early_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"No val Dice improvement greater than {early_min_delta} "
                f"for {early_patience} epochs."
            )
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "epoch": history_rows[-1]["epoch"],
            "best_val_dice": best_val_dice,
        },
        last_checkpoint_path,
    )

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 100)
    print(f"Loaded best checkpoint: {best_checkpoint_path}")
    print(f"Best val dice: {best_val_dice:.4f}")
    print("=" * 100)

    threshold_sweep_cfg = model_cfg.get("threshold_sweep", {})
    final_threshold = threshold

    val_records = collect_probabilities_for_split(
        model=model,
        dataloader=val_loader,
        dataset_root=dataset_root,
        split_cfg=val_cfg,
        device=device,
    )

    if threshold_sweep_cfg.get("enabled", False):
        threshold_values = threshold_sweep_cfg.get(
            "values",
            [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        )

        threshold_df = run_threshold_sweep(
            records=val_records,
            thresholds=threshold_values,
            postprocessing_cfg=postprocessing_cfg,
            #Classes added to support multiclass
            classes=classes,
        )

        save_threshold_sweep(threshold_df, output_root)

        choose_metric = threshold_sweep_cfg.get("choose_metric", "dice")

        if choose_metric not in threshold_df.columns:
            raise ValueError(
                f"choose_metric not found in threshold sweep: {choose_metric}"
            )

        best_row = threshold_df.sort_values(
            by=[choose_metric, "dice"],
            ascending=False,
        ).iloc[0]

        best_threshold = float(best_row["threshold"])

        print(
            f"Best validation threshold by {choose_metric}: "
            f"{best_threshold:.2f}"
        )

        if threshold_sweep_cfg.get("use_best_threshold_for_final_eval", False):
            final_threshold = best_threshold

    print("=" * 100)
    print(f"Final evaluation threshold: {final_threshold:.2f}")
    print("Final evaluation with saved masks and overlays")
    print("=" * 100)

    val_inference_time = estimate_inference_time_per_image(
        model=model,
        dataloader=val_loader,
        device=device,
    )

    test_inference_time = estimate_inference_time_per_image(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    evaluate_records_and_save(
        records=val_records,
        dataset_name=dataset_name,
        model_name=model_name,
        output_root=output_root,
        split_name="val",
        threshold=final_threshold,
        postprocessing_cfg=postprocessing_cfg,
        save_cfg=save_cfg,
        inference_time_ms_per_image=val_inference_time,
        classes=classes,
    )

    test_records = collect_probabilities_for_split(
        model=model,
        dataloader=test_loader,
        dataset_root=dataset_root,
        split_cfg=test_cfg,
        device=device,
    )

    evaluate_records_and_save(
        records=test_records,
        dataset_name=dataset_name,
        model_name=model_name,
        output_root=output_root,
        split_name="test",
        threshold=final_threshold,
        postprocessing_cfg=postprocessing_cfg,
        save_cfg=save_cfg,
        inference_time_ms_per_image=test_inference_time,
        classes=classes
    )

    print(f"{model_name} training and evaluation finished.")
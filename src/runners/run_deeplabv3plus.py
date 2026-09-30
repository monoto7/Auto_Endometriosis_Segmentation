from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.segmentation_dataset import BinarySegmentationDataset
#Multiclass Grayscale support
from src.datasets.segmentation_dataset import GrayscaleSegmentationDataset
from src.models.deeplabv3plus.deeplabv3plus_model import build_deeplabv3plus_model
from src.runners.run_unetpp import (
    collect_probabilities_for_split,
    estimate_inference_time_per_image,
    evaluate_records_and_save,
    run_threshold_sweep,
    save_threshold_sweep,
)

#Standardize mask handling for multiclass case
from src.utils.mask_utils import probability_to_mask
from src.utils.mask_utils import load_mask

#Standardize loss handling for multiclass case
from src.utils.loss_funcs import DiceCELossModule
from src.utils.loss_funcs import dice_loss_from_logits_multiclass

def load_yaml(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)

        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()

        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        return total_loss


def get_batch_images_masks(batch):
    if isinstance(batch, dict):
        image_keys = ["image", "images", "input", "pixel_values"]
        mask_keys = ["mask", "masks", "label", "labels"]

        images = None
        masks = None

        for key in image_keys:
            if key in batch:
                images = batch[key]
                break

        for key in mask_keys:
            if key in batch:
                masks = batch[key]
                break

        if images is None or masks is None:
            raise KeyError(f"Could not find image/mask keys in batch. Available keys: {batch.keys()}")

        return images, masks

    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def ensure_mask_shape(masks):
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)

    return masks.float()


def compute_batch_dice_from_logits(logits, masks, threshold: float = 0.5, smooth: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    masks_flat = masks.view(masks.size(0), -1)

    intersection = (preds_flat * masks_flat).sum(dim=1)
    denominator = preds_flat.sum(dim=1) + masks_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return dice.mean().item()


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0

    progress = tqdm(dataloader, desc="Train", leave=False)

    for batch in progress:
        images, masks = get_batch_images_masks(batch)

        images = images.to(device, non_blocking=True).float()
        masks = ensure_mask_shape(masks).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        #TODO see if this impacts multiclass setting
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, threshold: float = 0.5, binary = True):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Val", leave=False)

    for batch in progress:
        images, masks = get_batch_images_masks(batch)

        images = images.to(device, non_blocking=True).float()
        masks = ensure_mask_shape(masks).to(device, non_blocking=True)

        logits = model(images)

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = criterion(logits, masks)
        dice = None
        if(binary):
            dice = compute_batch_dice_from_logits(
                logits=logits,
                masks=masks,
                threshold=threshold,
            )
        else:
            dice = (1 - dice_loss_from_logits_multiclass(logits,masks.squeeze(1).long())).item()
        running_loss += loss.item() * images.size(0)
        running_dice += dice
        num_batches += 1

        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}",
            }
        )

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / max(num_batches, 1)

    return epoch_loss, epoch_dice


def save_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    epoch: int,
    best_val_dice: float,
    model_cfg: dict,
    exp_cfg: dict,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "model_cfg": model_cfg,
            "exp_cfg": exp_cfg,
        },
        checkpoint_path,
    )


def plot_training_curves(history_df: pd.DataFrame, output_root: Path):
    curves_dir = output_root / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    loss_curve_path = curves_dir / "loss_curve.png"

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    dice_curve_path = curves_dir / "val_dice_curve.png"

    plt.figure()
    plt.plot(history_df["epoch"], history_df["val_dice"], label="val_dice")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dice_curve_path, dpi=200)
    plt.close()

    print(f"Saved loss curve: {loss_curve_path}")
    print(f"Saved Dice curve: {dice_curve_path}")


def save_training_time_summary(
    output_root: Path,
    dataset_name: str,
    model_name: str,
    training_seconds: float,
    best_epoch: int,
    stopped_epoch: int,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
):
    summary_df = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "model_name": model_name,
                "training_state": "trained",
                "total_training_time_seconds": training_seconds,
                "total_training_time_minutes": training_seconds / 60.0,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
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


def train_and_evaluate(experiment_config_path):
    experiment_config_path = Path(experiment_config_path)
    project_root = Path(__file__).resolve().parents[2]

    exp_cfg = load_yaml(experiment_config_path)

    dataset_cfg = load_yaml(project_root / exp_cfg["dataset_config"])
    model_cfg = load_yaml(project_root / exp_cfg["model_config"])

    dataset_name = dataset_cfg["dataset_name"]
    dataset_root = Path(dataset_cfg["dataset_root"])
    output_root = Path(exp_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    model_name = model_cfg.get("model_name", "DeepLabV3Plus")

    train_split = exp_cfg["splits"]["train"]
    val_split = exp_cfg["splits"]["val"]
    test_split = exp_cfg["splits"]["test"]

    train_cfg = dataset_cfg["splits"][train_split]
    val_cfg = dataset_cfg["splits"][val_split]
    test_cfg = dataset_cfg["splits"][test_split]

    image_size = int(model_cfg.get("image_size", 512))
    batch_size = int(model_cfg.get("batch_size", 4))
    num_workers = int(model_cfg.get("num_workers", 4))

    #Allow experiment config to override model config
    num_labels = model_cfg.get("num_labels", 1)
    num_labels = exp_cfg.get("num_labels", num_labels)


    #Choose dataset loader based upon number of labels to prevent binarization
    Dataset = BinarySegmentationDataset

    if num_labels > 1:
        Dataset = GrayscaleSegmentationDataset
    classes = dataset_cfg.get("classes", ["255"])

    device = torch.device(
        model_cfg.get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset = Dataset(
        images_dir=dataset_root / train_cfg["images"],
        masks_dir=dataset_root / train_cfg["masks"],
        image_size=image_size,
        augment=True,
        classes=classes
    )

    val_dataset = Dataset(
        images_dir=dataset_root / val_cfg["images"],
        masks_dir=dataset_root / val_cfg["masks"],
        image_size=image_size,
        augment=False,
        classes=classes
    )

    test_dataset = Dataset(
        images_dir=dataset_root / test_cfg["images"],
        masks_dir=dataset_root / test_cfg["masks"],
        image_size=image_size,
        augment=False,
        classes=classes
    )

    #Dataloaders modified to drop last to prevent issues with batch sizes that produce a batch with a single entry at the end of an epoch
    #TODO test if this matters for testing/val
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    #Modified classes to use the actual number of labels
    model = build_deeplabv3plus_model(
        encoder_name=model_cfg.get("encoder_name", "resnet50"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=int(model_cfg.get("in_channels", 3)),
        classes=int(num_labels),
    )

    model.to(device)

    loss_cfg = model_cfg.get("loss", {})

    LossFunc = DiceBCELoss
    if(num_labels>1):
        LossFunc = DiceCELossModule

    #Removed explicit declaration of parameter names for simpler handling
    #TODO maybe should make a parent class both DiceBCELoss and DiceCELossModule inherit from
    criterion = LossFunc(
        float(loss_cfg.get("dice_weight", 0.5)),
        float(loss_cfg.get("bce_weight", 0.5)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("learning_rate", 0.0001)),
        weight_decay=float(model_cfg.get("weight_decay", 0.00001)),
    )

    scheduler_cfg = model_cfg.get("scheduler", {})

    if scheduler_cfg.get("use_reduce_on_plateau", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 8)),
            min_lr=float(scheduler_cfg.get("min_lr", 0.000001)),
        )
    else:
        scheduler = None

    epochs = int(model_cfg.get("epochs", 100))
    threshold = float(model_cfg.get("threshold", 0.5))

    early_cfg = model_cfg.get("early_stopping", {})
    use_early_stopping = bool(early_cfg.get("enabled", True))
    early_patience = int(early_cfg.get("patience", 20))
    min_delta = float(early_cfg.get("min_delta", 0.001))

    checkpoint_dir = output_root / "checkpoints"
    best_checkpoint_path = checkpoint_dir / f"best_{model_name}.pt"
    last_checkpoint_path = checkpoint_dir / f"last_{model_name}.pt"

    print("=" * 100)
    print(f"Training {model_name} on {dataset_name}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output root:  {output_root}")
    print(f"Device:       {device}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")
    print(f"Test images:  {len(test_dataset)}")
    print("=" * 100)

    history = []

    best_val_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    training_start_time = time.perf_counter()
    cumulative_time = 0.0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.perf_counter()

        print("\n" + "-" * 100)
        print(f"Epoch {epoch}/{epochs}")
        print("-" * 100)

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_dice = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            binary=num_labels==1
        )

        if scheduler is not None:
            scheduler.step(val_dice)

        epoch_time = time.perf_counter() - epoch_start_time
        cumulative_time += epoch_time

        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_dice > best_val_dice + min_delta

        if improved:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_dice=best_val_dice,
                model_cfg=model_cfg,
                exp_cfg=exp_cfg,
            )

            print(f"Saved new best checkpoint: {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_dice=best_val_dice,
            model_cfg=model_cfg,
            exp_cfg=exp_cfg,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "best_val_dice": best_val_dice,
                "best_epoch": best_epoch,
                "learning_rate": current_lr,
                "epoch_time_seconds": epoch_time,
                "cumulative_time_seconds": cumulative_time,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_dice={val_dice:.4f}, "
            f"best_val_dice={best_val_dice:.4f}, "
            f"lr={current_lr:.8f}, "
            f"time={epoch_time:.1f}s"
        )

        if use_early_stopping and epochs_without_improvement >= early_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch: {best_epoch}, best val Dice: {best_val_dice:.4f}"
            )
            break

    total_training_time = time.perf_counter() - training_start_time
    stopped_epoch = history[-1]["epoch"]

    history_df = pd.DataFrame(history)

    save_dataframe_csv_xlsx(
        df=history_df,
        csv_path=output_root / "training_history.csv",
        xlsx_path=output_root / "training_history.xlsx",
        sheet_name="training_history",
    )

    plot_training_curves(history_df, output_root)

    save_training_time_summary(
        output_root=output_root,
        dataset_name=dataset_name,
        model_name=model_name,
        training_seconds=total_training_time,
        best_epoch=best_epoch,
        stopped_epoch=stopped_epoch,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint_path}")

    print("=" * 100)
    print(f"Loading best checkpoint for final evaluation: {best_checkpoint_path}")
    print("=" * 100)

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    postprocessing_cfg = model_cfg.get("postprocessing", {})
    threshold_sweep_cfg = model_cfg.get("threshold_sweep", {})
    save_cfg = exp_cfg.get("save", {})

    #Appended to use classes derived from cfg
    #TODO Currently uses unetpp's implementation in it's runner file, this should be moved away from both into it's own util, was surprised when this initially happened.
    val_records = collect_probabilities_for_split(
        model=model,
        dataloader=val_loader,
        dataset_root=dataset_root,
        split_cfg=val_cfg,
        device=device,
        classes=classes
    )

    final_threshold = threshold

    if threshold_sweep_cfg.get("enabled", True):
        threshold_values = threshold_sweep_cfg.get(
            "values",
            [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        )
        #TODO Same as probabilities. should be moved into own shared util rather than using unetpps
        threshold_df = run_threshold_sweep(
            records=val_records,
            thresholds=threshold_values,
            postprocessing_cfg=postprocessing_cfg,
            classes=classes
        )

        save_threshold_sweep(threshold_df, output_root)

        if threshold_sweep_cfg.get("use_best_threshold_for_final_eval", False):
            choose_metric = threshold_sweep_cfg.get("choose_metric", "dice")

            best_row = threshold_df.sort_values(
                by=[choose_metric, "dice"],
                ascending=False,
            ).iloc[0]

            final_threshold = float(best_row["threshold"])

    print(f"Final evaluation threshold: {final_threshold:.2f}")

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
        classes=classes
    )

    test_records = collect_probabilities_for_split(
        model=model,
        dataloader=test_loader,
        dataset_root=dataset_root,
        split_cfg=test_cfg,
        device=device,
        classes=classes
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

    print(f"{model_name} training and evaluation finished for {dataset_name}.")
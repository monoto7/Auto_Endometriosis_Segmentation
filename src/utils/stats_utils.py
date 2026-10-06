import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.evaluation.metrics import compute_multiclass_metrics
from src.utils.mask_utils import probability_to_mask
from src.utils.mask_utils import save_mask
from src.utils.visualization import save_overlay


from pathlib import Path

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

        pred_mask = probability_to_mask(
            probability_map=record["probability_map"],
            threshold=threshold,
            postprocessing_cfg=postprocessing_cfg,
            classes=classes
        )

        merged_name = f"{Path(image_name).stem}.png"
        merged_path = merged_dir / merged_name

        if save_cfg.get("merged_masks", True):
            save_mask(pred_mask, merged_path)

        if save_cfg.get("overlays", True):
            overlay_path = overlay_dir / f"{Path(image_name).stem}_overlay.png"

            save_overlay(
                image_path=image_path,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                output_path=overlay_path,
            )

        for class_name in classes:
            #Use multiclass metrics and just assume 255, assuming pre-processing will make everything a binary mask in the non-multiclass case
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
                    "class_id": class_name,
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
                "class_id": class_name,
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

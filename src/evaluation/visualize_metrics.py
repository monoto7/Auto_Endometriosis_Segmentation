from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from openpyxl.styles import Font


# ============================================================
# Generic metric visualization and summary tables
# Works for models with different available prompt modes.
# ============================================================

DEFAULT_RESULTS_ROOT = Path(r"F:\Results\SAM_Benchmarking")

PREFERRED_PROMPT_ORDER = [
    "GT_point",
    "GT_box",
    "GT_box_point",
    "GT_box_posneg",
    "No_prompt",
    "Auto_box",
    "Auto_mask",
]

PROMPT_LABELS = {
    "GT_point": "GT point",
    "GT_box": "GT box",
    "GT_box_point": "GT box + point",
    "GT_box_posneg": "GT box + point + neg points",
    "No_prompt": "No prompt",
    "Auto_box": "Automatic box",
    "Auto_mask": "Automatic mask",
}

METRICS_TO_PLOT = [
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

RATIO_METRICS = {
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
}


def metric_display_name(metric: str) -> str:
    names = {
        "dice": "Dice score",
        "iou": "IoU",
        "precision": "Precision",
        "recall": "Recall / sensitivity",
        "specificity": "Specificity",
        "false_positive_area_px": "False-positive area (pixels)",
        "false_negative_area_px": "False-negative area (pixels)",
        "gt_area_px": "Ground-truth area (pixels)",
        "pred_area_px": "Predicted area (pixels)",
        "inference_time_ms": "Inference time (ms)",
        "sam_score": "SAM score",
        "selected_mask_index": "Selected mask index",
        "num_prompt_instances": "Number of prompt instances",
        "medsam_mean_probability": "MedSAM mean probability",
    }
    return names.get(metric, metric)


def detect_available_prompt_modes(
    results_root: Path,
    dataset_name: str,
    model_name: str,
    training_state: str,
    split: str,
):
    """
    Auto-detect prompt/result folders that contain metrics_image_level.csv.

    Example:
      F:/Results/SAM_Benchmarking/ENID/MedSAM/frozen/GT_box/val/metrics_image_level.csv
    """

    base_dir = results_root / dataset_name / model_name / training_state

    if not base_dir.exists():
        raise FileNotFoundError(f"Results folder not found: {base_dir}")

    detected = []

    for child in base_dir.iterdir():
        if not child.is_dir():
            continue

        if child.name == "visualize_metrics":
            continue

        metrics_csv = child / split / "metrics_image_level.csv"

        if metrics_csv.exists():
            detected.append(child.name)

    ordered = []

    for prompt_mode in PREFERRED_PROMPT_ORDER:
        if prompt_mode in detected:
            ordered.append(prompt_mode)

    for prompt_mode in sorted(detected):
        if prompt_mode not in ordered:
            ordered.append(prompt_mode)

    if not ordered:
        raise RuntimeError(
            f"No prompt/result folders with metrics_image_level.csv found for "
            f"{dataset_name}/{model_name}/{training_state}/{split}"
        )

    return ordered


def collect_csvs(
    results_root: Path,
    dataset_name: str,
    model_name: str,
    training_state: str,
    split: str,
):
    base_dir = results_root / dataset_name / model_name / training_state

    prompt_modes = detect_available_prompt_modes(
        results_root=results_root,
        dataset_name=dataset_name,
        model_name=model_name,
        training_state=training_state,
        split=split,
    )

    print(f"Detected result modes for {dataset_name} | {model_name} | {split}: {prompt_modes}")

    image_metric_dfs = []
    inference_dfs = []

    for prompt_mode in prompt_modes:
        prompt_dir = base_dir / prompt_mode / split

        metrics_csv = prompt_dir / "metrics_image_level.csv"
        inference_csv = prompt_dir / "inference_results.csv"

        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            df["dataset"] = dataset_name
            df["model_name"] = model_name
            df["training_state"] = training_state
            df["split"] = split
            df["prompt_mode"] = prompt_mode
            df["prompt_label"] = PROMPT_LABELS.get(prompt_mode, prompt_mode)
            df["source_table"] = "metrics_image_level"
            df["source_csv"] = str(metrics_csv)
            image_metric_dfs.append(df)

        if inference_csv.exists():
            df = pd.read_csv(inference_csv)
            df["dataset"] = dataset_name
            df["model_name"] = model_name
            df["training_state"] = training_state
            df["split"] = split
            df["prompt_mode"] = prompt_mode
            df["prompt_label"] = PROMPT_LABELS.get(prompt_mode, prompt_mode)
            df["source_table"] = "inference_results"
            df["source_csv"] = str(inference_csv)
            inference_dfs.append(df)

    if not image_metric_dfs:
        raise RuntimeError(
            f"No metrics_image_level.csv files found for "
            f"{dataset_name}/{model_name}/{training_state}/{split}"
        )

    image_metrics_df = pd.concat(image_metric_dfs, ignore_index=True)

    if inference_dfs:
        inference_df = pd.concat(inference_dfs, ignore_index=True)
    else:
        inference_df = pd.DataFrame()

    return image_metrics_df, inference_df, prompt_modes


def make_prompt_comparison_boxplot(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    training_state: str,
    split: str,
    prompt_modes: list,
    class_id: str="",
) -> Path:
    data = []
    labels = []

    for prompt_mode in prompt_modes:
        values = df.loc[df["prompt_mode"] == prompt_mode, metric].dropna().values

        if len(values) == 0:
            continue

        data.append(values)
        labels.append(PROMPT_LABELS.get(prompt_mode, prompt_mode))

    if not data:
        raise RuntimeError(f"No valid data found for metric: {metric}")

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.28,
        medianprops={
            "color": "orange",
            "linewidth": 2.8,
        },
        meanprops={
            "color": "red",
            "linestyle": "-",
            "linewidth": 2.4,
        },
        boxprops={
            "linewidth": 1.2,
            "color": "black",
        },
        whiskerprops={
            "linewidth": 1.1,
            "color": "black",
        },
        capprops={
            "linewidth": 1.1,
            "color": "black",
        },
        flierprops={
            "marker": "o",
            "markersize": 3,
            "alpha": 0.45,
            "markerfacecolor": "black",
            "markeredgecolor": "black",
        },
    )

    for patch in box["boxes"]:
        patch.set_facecolor("#BFDDF2")

    title = (
        f"{dataset_name}{class_id} | {model_name} {training_state} | "
        f"{split} | {metric_display_name(metric)}"
    )

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("Prompt / mode", fontsize=16)
    ax.set_ylabel(metric_display_name(metric), fontsize=16)

    ax.tick_params(axis="x", labelsize=14, rotation=15)
    ax.tick_params(axis="y", labelsize=14)

    if metric in RATIO_METRICS:
        ax.set_ylim(0, 1.05)

    ax.grid(axis="y", alpha=0.25)

    ax.plot([], [], color="red", linewidth=2.4, label="Mean")
    ax.plot([], [], color="orange", linewidth=2.8, label="Median")
    ax.legend(fontsize=13, frameon=False, loc="best")

    fig.tight_layout()

    out_name = f"{dataset_name}_{metric}_{split}_{class_id}_prompt_compare.png"
    out_path = output_dir / out_name

    fig.savefig(out_path, dpi=500)
    plt.close(fig)

    return out_path


def summarize_numeric_columns(
    df: pd.DataFrame,
    source_table_name: str,
    prompt_modes: list,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    excluded_numeric = {
        "lesion_id",
    }

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [column for column in numeric_cols if column not in excluded_numeric]

    rows = []

    for parameter in numeric_cols:
        for prompt_mode in prompt_modes:
            sub = df.loc[df["prompt_mode"] == prompt_mode, parameter].dropna()

            if len(sub) == 0:
                continue

            rows.append(
                {
                    "parameter": parameter,
                    "parameter_display_name": metric_display_name(parameter),
                    "source_table": source_table_name,
                    "prompt_mode": prompt_mode,
                    "prompt_label": PROMPT_LABELS.get(prompt_mode, prompt_mode),
                    "n": int(len(sub)),
                    "min": float(sub.min()),
                    "max": float(sub.max()),
                    "Q1": float(sub.quantile(0.25)),
                    "Q3": float(sub.quantile(0.75)),
                    "median": float(sub.median()),
                    "mean": float(sub.mean()),
                    "std": float(sub.std()) if len(sub) > 1 else 0.0,
                }
            )

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df = summary_df[
            [
                "parameter",
                "parameter_display_name",
                "source_table",
                "prompt_mode",
                "prompt_label",
                "n",
                "min",
                "max",
                "Q1",
                "Q3",
                "median",
                "mean",
                "std",
            ]
        ]

    return summary_df


def save_excel_summary(
    image_metrics_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    split: str,
    prompt_modes: list,
    class_id: str = "",
) -> Path:
    image_summary = summarize_numeric_columns(
        image_metrics_df,
        source_table_name="metrics_image_level",
        prompt_modes=prompt_modes,
    )

    inference_summary = summarize_numeric_columns(
        inference_df,
        source_table_name="inference_results",
        prompt_modes=prompt_modes,
    )

    summary_all = pd.concat(
        [image_summary, inference_summary],
        ignore_index=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    out_xlsx = output_dir / f"{dataset_name}_{split}_{class_id}_summary_statistics.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_all.to_excel(writer, sheet_name="summary_all", index=False)
        image_summary.to_excel(writer, sheet_name="image_level_summary", index=False)

        if not inference_summary.empty:
            inference_summary.to_excel(writer, sheet_name="inference_summary", index=False)

        image_metrics_df.to_excel(writer, sheet_name="image_level_raw", index=False)

        if not inference_df.empty:
            inference_df.to_excel(writer, sheet_name="inference_raw", index=False)

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]

            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:
                    value_length = len(str(cell.value)) if cell.value is not None else 0
                    max_length = max(max_length, value_length)

                adjusted_width = min(max(max_length + 2, 10), 35)
                worksheet.column_dimensions[column_letter].width = adjusted_width

            worksheet.freeze_panes = "A2"

            for cell in worksheet[1]:
                cell.font = Font(bold=True)

    return out_xlsx


def visualize_dataset_model_split(
    results_root: Path,
    dataset_name: str,
    model_name: str,
    training_state: str,
    split: str,
):
    print("\n" + "=" * 100)
    print(f"Visualizing: {dataset_name} | {model_name} | {training_state} | {split}")
    print("=" * 100)

    image_metrics_df, inference_df, prompt_modes = collect_csvs(
        results_root=results_root,
        dataset_name=dataset_name,
        model_name=model_name,
        training_state=training_state,
        split=split,
    )

    output_dir = (
        results_root
        / dataset_name
        / model_name
        / training_state
        / "visualize_metrics"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    class_ids = []
    if 'class_id' in image_metrics_df:
        class_ids = image_metrics_df['class_id'].unique()

    
    
    for id in class_ids:
        combined_csv_class = output_dir / f"{dataset_name}_{split}_{id}_combined_prompt_metrics.csv"
        image_metrics_df.loc[image_metrics_df['class_id'] == id].to_csv(combined_csv_class, index=False)
        print(f"Saved combined image-level class {id} CSV: {combined_csv_class}")

    combined_csv = output_dir / f"{dataset_name}_{split}_combined_prompt_metrics.csv"
    image_metrics_df.to_csv(combined_csv, index=False)
    print(f"Saved combined image-level CSV: {combined_csv}")

    if not inference_df.empty:
        for id in class_ids:
            combined_inference_csv_class = output_dir / f"{dataset_name}_{split}_{id}_combined_inference_results.csv"
            inference_df.loc[inference_df['class_id'] == id].to_csv(combined_inference_csv_class, index=False)
            print(f"Saved combined inference class {id} CSV: {combined_inference_csv_class}")
        combined_inference_csv = output_dir / f"{dataset_name}_{split}_combined_inference_results.csv"
        inference_df.to_csv(combined_inference_csv, index=False)
        print(f"Saved combined inference CSV: {combined_inference_csv}")


    for id in class_ids:
        excel_path = save_excel_summary(
            image_metrics_df=image_metrics_df.loc[image_metrics_df['class_id'] == id],
            inference_df=inference_df.loc[inference_df['class_id'] == id],
            output_dir=output_dir,
            dataset_name=dataset_name,
            split=split,
            prompt_modes=prompt_modes,
            class_id=id
        )
        print(f"Saved Excel summary for {id}: {excel_path}")
    excel_path = save_excel_summary(
        image_metrics_df=image_metrics_df,
        inference_df=inference_df,
        output_dir=output_dir,
        dataset_name=dataset_name,
        split=split,
        prompt_modes=prompt_modes,
    )
    print(f"Saved Excel summary: {excel_path}")


    for id in class_ids:
        for metric in METRICS_TO_PLOT:
            if metric not in image_metrics_df.columns:
                print(f"WARNING: metric not found in image-level CSVs, skipping plot: {metric}")
                continue

            out_path = make_prompt_comparison_boxplot(
                df=image_metrics_df.loc[image_metrics_df['class_id'] == id],
                metric=metric,
                output_dir=output_dir,
                dataset_name=dataset_name,
                model_name=model_name,
                training_state=training_state,
                split=split,
                prompt_modes=prompt_modes,
                class_id=id
            )

            print(f"Saved plot: {out_path}")
    for metric in METRICS_TO_PLOT:
        if metric not in image_metrics_df.columns:
            print(f"WARNING: metric not found in image-level CSVs, skipping plot: {metric}")
            continue

        out_path = make_prompt_comparison_boxplot(
            df=image_metrics_df,
            metric=metric,
            output_dir=output_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            training_state=training_state,
            split=split,
            prompt_modes=prompt_modes,
        )

        print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ENID", "GLENDA"],
    )

    parser.add_argument(
        "--model",
        type=str,
        default="SAM2",
    )

    parser.add_argument(
        "--training_state",
        type=str,
        default="frozen",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
    )

    args = parser.parse_args()

    results_root = Path(args.results_root)

    for dataset_name in args.datasets:
        for split in args.splits:
            visualize_dataset_model_split(
                results_root=results_root,
                dataset_name=dataset_name,
                model_name=args.model,
                training_state=args.training_state,
                split=split,
            )

    print("\nDone visualizing metrics.")


if __name__ == "__main__":
    main()
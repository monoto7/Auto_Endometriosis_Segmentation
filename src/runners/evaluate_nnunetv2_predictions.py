from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_multiclass_metrics
from src.utils.visualization import save_overlay
from src.utils.nnunet_env import (
    setup_nnunet_environment,
    NNUNET_RAW,
    NNUNET_EXPORTS,
    NNUNET_RESULTS,
    NNUNET_EPOCH_STRING,
)

from src.utils.mask_utils import load_mask
from src.utils.mask_utils import save_mask
from src.utils.mask_utils import rescale_mask
from src.utils.misc_utils import load_yaml




RESULTS_ROOT = Path(r"/data/m_meyer/AES/OrganSegmentationResults/")

DATASETS = [
    # {
    #     "dataset_name": "ENID",
    #     "dataset_folder": "Dataset501_ENID",
    #     "output_name": "ENID",
    # },
    # {
    #     "dataset_name": "GLENDA",
    #     "dataset_folder": "Dataset502_GLENDA",
    #     "output_name": "GLENDA",
    # },
    # {
    #     "dataset_name": "GLENDA_clean",
    #     "dataset_folder": "Dataset503_GLENDA_clean",
    #     "output_name": "GLENDA_clean",
    # },
    # {
    #         "dataset_name": "GynSurg",
    #         "dataset_folder": "Dataset504_GynSurg",
    #         "output_name": "GynSurg",
    #         "dataset_conf": "gynsurg.yaml"
    # },
    # {
    #         "dataset_name": "GynSurg85",
    #         "dataset_folder": "Dataset505_GynSurg85",
    #         "output_name": "GynSurg85",
    # },
    # {
    #         "dataset_name": "GynSurg170",
    #         "dataset_folder": "Dataset506_GynSurg170",
    #         "output_name": "GynSurg170",
    # },
    {
            "dataset_name": "GynSurg255",
            "dataset_folder": "Dataset507_GynSurg255",
            "output_name": "GynSurg255",
    },
]

#MODEL_NAME = "nnUNetV2_2D"
MODEL_NAME = f"nnUNetV2_2D{NNUNET_EPOCH_STRING}"
TRAINING_STATE = "trained"
PROMPT_MODE = "No_prompt"

def save_dataframe_csv_xlsx(
    df: pd.DataFrame,
    csv_path: Path,
    xlsx_path: Path,
    sheet_name: str,
):
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
                60,
            )


def load_conversion_report(raw_dataset_folder: Path, dataset_name: str) -> pd.DataFrame:
    report_path = raw_dataset_folder / f"{dataset_name}_nnunet_conversion_report.csv"

    if not report_path.exists():
        raise FileNotFoundError(f"Conversion report not found: {report_path}")

    return pd.read_csv(report_path)


def find_prediction_path(prediction_folder: Path, case_id: str) -> Path:
    candidates = [
        prediction_folder / f"{case_id}.png",
        prediction_folder / f"{case_id}.nii.gz",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    png_matches = list(prediction_folder.glob(f"{case_id}*.png"))

    if png_matches:
        return png_matches[0]

    nii_matches = list(prediction_folder.glob(f"{case_id}*.nii.gz"))

    if nii_matches:
        return nii_matches[0]

    raise FileNotFoundError(
        f"Prediction not found for case_id={case_id} in {prediction_folder}"
    )


def evaluate_split(
    dataset_name: str,
    conversion_df: pd.DataFrame,
    prediction_folder: Path,
    output_root: Path,
    split_name: str,
    classes: list = ["255"],
):
    split_df = conversion_df[conversion_df["split"] == split_name].copy()

    out_dir = output_root / PROMPT_MODE / split_name
    merged_dir = out_dir / "merged_masks"
    overlay_dir = out_dir / "overlays"

    merged_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    inference_rows = []
    metric_rows = []

    for _, row in tqdm(
        split_df.iterrows(),
        total=len(split_df),
        desc=f"{dataset_name} {split_name}",
    ):
        case_id = row["case_id"]
        source_image = Path(row["source_image"])
        source_mask = Path(row["source_mask"])
        gt_label_path = Path(row["nnunet_label"])

        prediction_path = find_prediction_path(
            prediction_folder=prediction_folder,
            case_id=case_id,
        )

        gt_mask = load_mask(gt_label_path)
        pred_mask = load_mask(prediction_path)

        if pred_mask.shape != gt_mask.shape:
            pred_image = Image.fromarray(pred_mask.astype(np.uint8))
            pred_image = pred_image.resize(
                (gt_mask.shape[1], gt_mask.shape[0]),
                resample=Image.NEAREST,
            )
            pred_mask = np.array(pred_image).astype(np.uint8)

        merged_mask_name = f"{case_id}.png"
        merged_mask_path = merged_dir / merged_mask_name

        pred_mask = rescale_mask(pred_mask,classes=classes)
        gt_mask = rescale_mask(gt_mask, classes=classes)

        save_mask(pred_mask, merged_mask_path)

        overlay_path = overlay_dir / f"{case_id}_overlay.png"

        save_overlay(
            image_path=source_image,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            output_path=overlay_path,
        )
        if len(classes)>1 :
            for class_value in classes:
                overlay_path = overlay_dir / f"{case_id}_{class_value}_overlay.png"
                save_overlay(
                    image_path=source_image,
                    gt_mask=gt_mask==int(class_value),
                    pred_mask=pred_mask==int(class_value),
                    output_path=overlay_path,
                )

        for class_value in classes:
            metrics = compute_multiclass_metrics(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                class_g_value=int(class_value)
            )

            inference_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "model_name": MODEL_NAME,
                    "training_state": TRAINING_STATE,
                    "prompt_mode": PROMPT_MODE,
                    "image_name": source_image.name,
                    "mask_name": source_mask.name,
                    "case_id": case_id,
                    "class_id": class_value,
                    "prediction_name": prediction_path.name,
                    "merged_mask_name": merged_mask_name,
                    "inference_time_ms": None,
                }
            )

            metric_row = {
                "dataset": dataset_name,
                "split": split_name,
                "model_name": MODEL_NAME,
                "training_state": TRAINING_STATE,
                "prompt_mode": PROMPT_MODE,
                "image_name": source_image.name,
                "mask_name": source_mask.name,
                "case_id": case_id,
                "class_id": class_value,
                "num_prompt_instances": 0,
            }

            metric_row.update(metrics)
            metric_rows.append(metric_row)

    inference_df = pd.DataFrame(inference_rows)
    metrics_df = pd.DataFrame(metric_rows)

    save_dataframe_csv_xlsx(
        df=inference_df,
        csv_path=out_dir / "inference_results.csv",
        xlsx_path=out_dir / "inference_results.xlsx",
        sheet_name="inference_results",
    )

    save_dataframe_csv_xlsx(
        df=metrics_df,
        csv_path=out_dir / "metrics_image_level.csv",
        xlsx_path=out_dir / "metrics_image_level.xlsx",
        sheet_name="metrics_image_level",
    )

    numeric_cols = metrics_df.select_dtypes(include="number").columns

    summary_df = metrics_df[numeric_cols].agg(
        ["mean", "std", "median", "min", "max"]
    ).T.reset_index()

    summary_df = summary_df.rename(columns={"index": "metric"})

    save_dataframe_csv_xlsx(
        df=summary_df,
        csv_path=out_dir / "metrics_summary.csv",
        xlsx_path=out_dir / "metrics_summary.xlsx",
        sheet_name="metrics_summary",
    )

    print(f"Saved nnU-Net {split_name} outputs: {out_dir}")


def copy_checkpoint_summary(dataset_folder: str, output_root: Path):
    result_dataset_dir = NNUNET_RESULTS / dataset_folder

    rows = []

    for checkpoint_path in result_dataset_dir.rglob("checkpoint_best.pth"):
        rows.append(
            {
                "checkpoint_type": "best",
                "checkpoint_path": str(checkpoint_path),
            }
        )

    for checkpoint_path in result_dataset_dir.rglob("checkpoint_final.pth"):
        rows.append(
            {
                "checkpoint_type": "final",
                "checkpoint_path": str(checkpoint_path),
            }
        )

    if not rows:
        print(f"WARNING: No checkpoints found under {result_dataset_dir}")
        return

    df = pd.DataFrame(rows)

    save_dataframe_csv_xlsx(
        df=df,
        csv_path=output_root / "checkpoint_summary.csv",
        xlsx_path=output_root / "checkpoint_summary.xlsx",
        sheet_name="checkpoint_summary",
    )


def evaluate_dataset(dataset_cfg):
    dataset_name = dataset_cfg["dataset_name"]
    dataset_folder = dataset_cfg["dataset_folder"]
    output_name = dataset_cfg["output_name"]

    #Load dataset conf for proper multiclass output
    dataset_conf_loc = dataset_cfg.get("dataset_conf", None)

    dataset_conf = None
    classes = ["255"]
    if(dataset_conf_loc is not None):
        dataset_conf = load_yaml(PROJECT_ROOT / "configs" / "datasets" / dataset_conf_loc)
        classes = dataset_conf.get("classes", ["255"])
    raw_dataset_folder = NNUNET_RAW / dataset_folder

    conversion_df = load_conversion_report(
        raw_dataset_folder=raw_dataset_folder,
        dataset_name=dataset_name,
    )

    output_root = RESULTS_ROOT / output_name / MODEL_NAME / TRAINING_STATE
    output_root.mkdir(parents=True, exist_ok=True)

    all_images_tr_predictions = (
        NNUNET_EXPORTS
        / dataset_folder
        / f"all_imagesTr_predictions{NNUNET_EPOCH_STRING}"
    )

    test_predictions = (
        NNUNET_EXPORTS
        / dataset_folder
        / f"test_predictions{NNUNET_EPOCH_STRING}"
    )

    if not all_images_tr_predictions.exists():
        raise FileNotFoundError(f"Missing predictions: {all_images_tr_predictions}")

    if not test_predictions.exists():
        raise FileNotFoundError(f"Missing predictions: {test_predictions}")

    evaluate_split(
        dataset_name=dataset_name,
        conversion_df=conversion_df,
        prediction_folder=all_images_tr_predictions,
        output_root=output_root,
        split_name="val",
        classes=classes,
    )

    evaluate_split(
        dataset_name=dataset_name,
        conversion_df=conversion_df,
        prediction_folder=test_predictions,
        output_root=output_root,
        split_name="test",
        classes=classes,
    )

    copy_checkpoint_summary(
        dataset_folder=dataset_folder,
        output_root=output_root,
    )


def main():
    setup_nnunet_environment()

    for dataset_cfg in DATASETS:
        evaluate_dataset(dataset_cfg)


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


NNUNET_RAW = Path(r"C:\Users\cooll\OneDrive\Documents\SPARC\Preprocessing\nnUnet\nnUNet_raw")

DATASETS = [
    #"Dataset501_ENID",
    #"Dataset502_GLENDA",
    #"Dataset503_GLENDA_clean",
    "Dataset504_GynSurg"
]

LABEL_FOLDERS = [
    "labelsTr",
    "labelsTs_reference",
]

MAKE_PREVIEWS = True


def read_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    return np.array(mask)


def save_preview_mask(mask_array: np.ndarray, output_path: Path):
    """
    Saves a visual preview where all foreground labels become 255.

    This is only for inspection.
    Do not train nnU-Net on preview masks.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    preview = (mask_array > 0).astype(np.uint8) * 255
    Image.fromarray(preview).save(output_path)


def check_label_folder(dataset_folder: Path, label_folder_name: str):
    label_folder = dataset_folder / label_folder_name

    if not label_folder.exists():
        print(f"Missing folder: {label_folder}")
        return []

    preview_folder = dataset_folder / f"{label_folder_name}_preview_255"

    if MAKE_PREVIEWS:
        preview_folder.mkdir(parents=True, exist_ok=True)

    rows = []

    mask_paths = sorted(label_folder.glob("*.png"))

    if len(mask_paths) == 0:
        print(f"No PNG masks found in: {label_folder}")

    for mask_path in mask_paths:
        mask_array = read_mask(mask_path)

        unique_values = sorted(np.unique(mask_array).tolist())
        foreground_pixels = int((mask_array > 0).sum())
        total_pixels = int(mask_array.size)

        foreground_fraction = (
            foreground_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        min_value = int(mask_array.min())
        max_value = int(mask_array.max())

        is_empty = int(foreground_pixels == 0)
        has_invalid_values = int(
            not set(unique_values).issubset({0, 1})
        )

        preview_path = ""

        if MAKE_PREVIEWS:
            preview_output_path = preview_folder / mask_path.name
            save_preview_mask(
                mask_array=mask_array,
                output_path=preview_output_path,
            )
            preview_path = str(preview_output_path)

        rows.append(
            {
                "dataset": dataset_folder.name,
                "label_folder": label_folder_name,
                "mask_name": mask_path.name,
                "mask_path": str(mask_path),
                "preview_path": preview_path,
                "unique_values": str(unique_values),
                "min": min_value,
                "max": max_value,
                "foreground_pixels": foreground_pixels,
                "total_pixels": total_pixels,
                "foreground_fraction": foreground_fraction,
                "is_empty": is_empty,
                "has_invalid_values": has_invalid_values,
            }
        )

    return rows


def save_dataframe_csv_xlsx(df: pd.DataFrame, csv_path: Path, xlsx_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="mask_check", index=False)

        worksheet = writer.sheets["mask_check"]
        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter

            for cell in column_cells:
                value_length = len(str(cell.value)) if cell.value is not None else 0
                max_length = max(max_length, value_length)

            worksheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 10),
                70,
            )


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("Mask count and empty-mask summary")
    print("=" * 100)

    print(
        df.groupby(["dataset", "label_folder"])["is_empty"]
        .agg(["count", "sum"])
        .rename(columns={"count": "num_masks", "sum": "num_empty_masks"})
    )

    print("\n" + "=" * 100)
    print("Max value distribution")
    print("=" * 100)

    print(
        df.groupby(["dataset", "label_folder"])["max"]
        .value_counts()
        .rename("count")
    )

    print("\n" + "=" * 100)
    print("Invalid-value summary")
    print("=" * 100)

    print(
        df.groupby(["dataset", "label_folder"])["has_invalid_values"]
        .sum()
        .rename("num_invalid_masks")
    )

    print("\n" + "=" * 100)
    print("Foreground fraction summary")
    print("=" * 100)

    print(
        df.groupby(["dataset", "label_folder"])["foreground_fraction"]
        .agg(["mean", "median", "min", "max"])
    )


def main():
    all_rows = []

    for dataset_name in DATASETS:
        dataset_folder = NNUNET_RAW / dataset_name

        if not dataset_folder.exists():
            print(f"Missing dataset folder: {dataset_folder}")
            continue

        for label_folder_name in LABEL_FOLDERS:
            rows = check_label_folder(
                dataset_folder=dataset_folder,
                label_folder_name=label_folder_name,
            )
            all_rows.extend(rows)

    if len(all_rows) == 0:
        print("No masks found.")
        return

    df = pd.DataFrame(all_rows)

    print_summary(df)

    output_csv = NNUNET_RAW / "nnunet_mask_check.csv"
    output_xlsx = NNUNET_RAW / "nnunet_mask_check.xlsx"

    save_dataframe_csv_xlsx(
        df=df,
        csv_path=output_csv,
        xlsx_path=output_xlsx,
    )

    print("\n" + "=" * 100)
    print("Saved outputs")
    print("=" * 100)
    print(f"Mask check CSV:  {output_csv}")
    print(f"Mask check XLSX: {output_xlsx}")

    if MAKE_PREVIEWS:
        print("\nPreview masks saved under each dataset folder:")
        for dataset_name in DATASETS:
            dataset_folder = NNUNET_RAW / dataset_name
            for label_folder_name in LABEL_FOLDERS:
                preview_folder = dataset_folder / f"{label_folder_name}_preview_255"
                if preview_folder.exists():
                    print(preview_folder)


if __name__ == "__main__":
    main()
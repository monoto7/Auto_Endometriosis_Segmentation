from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.nnunet_env import (
    setup_nnunet_environment,
    NNUNET_RAW,
    NNUNET_PREPROCESSED,
)


DATASETS = [
    {
        "dataset_id": 501,
        "dataset_name": "ENID",
        "standardized_root": Path(
            r"F:\Datasets\Standardized datasets\ENID\ENID 60_20_20 Split"
        ),
        "nnunet_name": "ENID",
    },
    {
        "dataset_id": 502,
        "dataset_name": "GLENDA",
        "standardized_root": Path(
            r"F:\Datasets\Standardized datasets\GLENDA\GLENDA 60_20_20 split"
        ),
        "nnunet_name": "GLENDA",
    },
    {
        "dataset_id": 503,
        "dataset_name": "GLENDA_clean",
        "standardized_root": Path(
            r"F:\Datasets\Standardized datasets\GLENDA_clean\GLENDA_clean 60_20_20 split"
        ),
        "nnunet_name": "GLENDA_clean",
    },
    {
        "dataset_id": 504,
        "dataset_name": "GynSurg",
        "standardized_root": Path(
            r"C:\\Users\\cooll\\OneDrive\\Documents\\SPARC\\SplitDatasets\\GynSurg"
        ),
        "nnunet_name": "GynSurg",
        #Added to have handling for multiclass cases
        "labels": {
            "background": 0,
            "uterus": 1,
            "fallopian tubes": 2,
            "ovaries": 3,
        },
    },
]

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def find_files(folder: Path):
    files = []

    for extension in IMAGE_EXTENSIONS:
        files.extend(sorted(folder.glob(f"*{extension}")))

    return sorted(files)


def find_matching_mask(masks_dir: Path, image_path: Path) -> Path:
    stem = image_path.stem

    for extension in IMAGE_EXTENSIONS:
        candidate = masks_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No matching mask found for image: {image_path}")


def read_rgb_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def read_label_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    # nnU-Net label map:
    # 0 = background
    # 1 = lesion
    label_np = (mask_np > 0).astype(np.uint8)

    unique_values = set(np.unique(label_np).tolist())

    if not unique_values.issubset({0, 1}):
        raise ValueError(f"Invalid label values in {mask_path}: {unique_values}")

    return label_np

def read_label_mask_multiclass(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    label_np = np.zeros(mask_np.shape)
    for i,val in enumerate(np.unique(mask_np)):
        label_np[mask_np==val] = i

    #Below commented out under the assumption it's not necessary for functionality
    #unique_values = set(np.unique(label_np).tolist())

    #if not unique_values.issubset({0, 1}):
    #    raise ValueError(f"Invalid label values in {mask_path}: {unique_values}")

    return label_np


def save_rgb_png(image_np: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np.astype(np.uint8)).save(output_path)


def save_label_png(label_np: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_np.astype(np.uint8)).save(output_path)


def clean_dataset_folder(dataset_folder: Path):
    if dataset_folder.exists():
        shutil.rmtree(dataset_folder)

    dataset_folder.mkdir(parents=True, exist_ok=True)


def make_case_id(dataset_name: str, split: str, index: int) -> str:
    safe_name = dataset_name.replace("-", "_").replace(" ", "_")
    return f"{safe_name}_{split}_{index:06d}"


def convert_split_to_nnunet(
    dataset_name: str,
    standardized_root: Path,
    dataset_folder: Path,
    split: str,
    multiclass: bool = False,
):
    images_dir = standardized_root / split / "images"
    masks_dir = standardized_root / split / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    image_paths = find_files(images_dir)

    rows = []
    case_ids = []

    for index, image_path in enumerate(
        tqdm(image_paths, desc=f"{dataset_name} {split}"),
        start=1,
    ):
        mask_path = find_matching_mask(masks_dir, image_path)

        image_np = read_rgb_image(image_path)

        mask_reader = read_label_mask
        if(multiclass):
            mask_reader = read_label_mask_multiclass
        label_np = mask_reader(mask_path)

        if image_np.shape[:2] != label_np.shape[:2]:
            raise ValueError(
                f"Image/mask shape mismatch for {image_path.name}: "
                f"image={image_np.shape[:2]}, mask={label_np.shape[:2]}"
            )

        case_id = make_case_id(
            dataset_name=dataset_name,
            split=split,
            index=index,
        )

        if split in ["train", "val"]:
            nnunet_image_path = dataset_folder / "imagesTr" / f"{case_id}_0000.png"
            nnunet_label_path = dataset_folder / "labelsTr" / f"{case_id}.png"
        elif split == "test":
            nnunet_image_path = dataset_folder / "imagesTs" / f"{case_id}_0000.png"
            nnunet_label_path = dataset_folder / "labelsTs_reference" / f"{case_id}.png"
        else:
            raise ValueError(f"Unsupported split: {split}")

        save_rgb_png(image_np, nnunet_image_path)
        save_label_png(label_np, nnunet_label_path)

        foreground_pixels = int(label_np.sum())

        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "case_id": case_id,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "nnunet_image": str(nnunet_image_path),
                "nnunet_label": str(nnunet_label_path),
                "height": int(label_np.shape[0]),
                "width": int(label_np.shape[1]),
                "foreground_pixels": foreground_pixels,
                "has_foreground": int(foreground_pixels > 0),
            }
        )

        case_ids.append(case_id)

    return rows, case_ids


def write_dataset_json(dataset_folder: Path, num_training: int):
    """
    Correct for RGB PNGs read by NaturalImage2DIO.

    The previous version used {"0": "RGB"}, which told nnU-Net to expect 1 channel.
    NaturalImage2DIO reads RGB PNGs as 3 channels, so we declare red/green/blue.
    """

    dataset_json = {
        "channel_names": {
            "0": "red",
            "1": "green",
            "2": "blue"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": num_training,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO"
    }

    output_path = dataset_folder / "dataset.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(dataset_json, file, indent=4)

    print(f"Saved dataset.json: {output_path}")

def write_dataset_json_multiclass(dataset_folder: Path, num_training: int, labels):
    """
    Correct for RGB PNGs read by NaturalImage2DIO.

    The previous version used {"0": "RGB"}, which told nnU-Net to expect 1 channel.
    NaturalImage2DIO reads RGB PNGs as 3 channels, so we declare red/green/blue.

    TODO
    to be reworked so labels are not hardcoded
    """

    dataset_json = {
        "channel_names": {
            "0": "red",
            "1": "green",
            "2": "blue"
        },
        #"labels": {
        #    "background": 0,
        #    "uterus": 1,
        #    "fallopian tubes": 2,
        #    "ovaries": 3,
        #},
        "labels" : labels,
        "numTraining": num_training,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO"
    }

    output_path = dataset_folder / "dataset.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(dataset_json, file, indent=4)

    print(f"Saved dataset.json: {output_path}")



def write_custom_split_file(dataset_folder_name: str, train_ids, val_ids):
    preprocessed_dataset_dir = NNUNET_PREPROCESSED / dataset_folder_name
    preprocessed_dataset_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        {
            "train": train_ids,
            "val": val_ids,
        }
    ]

    split_path = preprocessed_dataset_dir / "splits_final.json"

    with open(split_path, "w", encoding="utf-8") as file:
        json.dump(splits, file, indent=4)

    print(f"Saved custom fold split: {split_path}")


def save_conversion_report(dataset_folder: Path, dataset_name: str, rows):
    report_df = pd.DataFrame(rows)

    csv_path = dataset_folder / f"{dataset_name}_nnunet_conversion_report.csv"
    xlsx_path = dataset_folder / f"{dataset_name}_nnunet_conversion_report.xlsx"

    report_df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        report_df.to_excel(writer, sheet_name="conversion_report", index=False)

        worksheet = writer.sheets["conversion_report"]
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

    print(f"Saved conversion report CSV:  {csv_path}")
    print(f"Saved conversion report XLSX: {xlsx_path}")


def convert_dataset(dataset_cfg):
    dataset_id = int(dataset_cfg["dataset_id"])
    dataset_name = dataset_cfg["dataset_name"]
    nnunet_name = dataset_cfg["nnunet_name"]
    standardized_root = dataset_cfg["standardized_root"]
    labels = dataset_cfg.get(["labels"])

    dataset_folder_name = f"Dataset{dataset_id:03d}_{nnunet_name}"
    dataset_folder = NNUNET_RAW / dataset_folder_name

    print("=" * 100)
    print(f"Preparing nnU-Net v2 dataset: {dataset_folder_name}")
    print(f"Source: {standardized_root}")
    print(f"Target: {dataset_folder}")
    print("=" * 100)

    clean_dataset_folder(dataset_folder)

    all_rows = []

    train_rows, train_ids = convert_split_to_nnunet(
        dataset_name=dataset_name,
        standardized_root=standardized_root,
        dataset_folder=dataset_folder,
        split="train",
        multiclass=(labels is None)
    )

    val_rows, val_ids = convert_split_to_nnunet(
        dataset_name=dataset_name,
        standardized_root=standardized_root,
        dataset_folder=dataset_folder,
        split="val",
        multiclass=(labels is None)
    )

    test_rows, test_ids = convert_split_to_nnunet(
        dataset_name=dataset_name,
        standardized_root=standardized_root,
        dataset_folder=dataset_folder,
        split="test",
        multiclass=(labels is None)
    )

    all_rows.extend(train_rows)
    all_rows.extend(val_rows)
    all_rows.extend(test_rows)

    num_training = len(train_ids) + len(val_ids)

    if(labels is None):
        write_dataset_json(
            dataset_folder=dataset_folder,
            num_training=num_training,
        )
    else:
        write_dataset_json_multiclass(
            dataset_folder=dataset_folder,
            num_training=num_training,
            labels=labels
        )

    write_custom_split_file(
        dataset_folder_name=dataset_folder_name,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    save_conversion_report(
        dataset_folder=dataset_folder,
        dataset_name=dataset_name,
        rows=all_rows,
    )

    report_df = pd.DataFrame(all_rows)

    print("=" * 100)
    print(f"Finished {dataset_folder_name}")
    print("Cases per split:")
    print(report_df.groupby("split")["case_id"].count())
    print("Foreground cases per split:")
    print(report_df.groupby("split")["has_foreground"].sum())
    print("=" * 100)


def main():
    setup_nnunet_environment()

    NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)

    for dataset_cfg in DATASETS:
        convert_dataset(dataset_cfg)


if __name__ == "__main__":
    main()
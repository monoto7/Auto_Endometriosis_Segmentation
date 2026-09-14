from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

'''
his conversion treats connected components in your binary mask as pseudo-instances.
If two lesions touch, YOLO will see them as one instance.
'''
DATASETS = [
    {
        "dataset_name": "ENID",
        "standardized_root": Path(
            r"F:\Datasets\Standardized datasets\ENID\ENID 60_20_20 Split"
        ),
        "yolo_root": Path(r"F:\Datasets\YOLO datasets\ENID_yolo_seg"),
        "yaml_name": "enid_yolo_seg.yaml",
    },
    {
        "dataset_name": "GLENDA",
        "standardized_root": Path(
            r"F:\Datasets\Standardized datasets\GLENDA\GLENDA 60_20_20 split"
        ),
        "yolo_root": Path(r"F:\Datasets\YOLO datasets\GLENDA_yolo_seg"),
        "yaml_name": "glenda_yolo_seg.yaml",
    },
    {
        "dataset_name": "GLENDA_clean",
        "standardized_root": Path(
            r"F:\Datasets\Standardized datasets\GLENDA_clean\GLENDA_clean 60_20_20 split"
        ),
        "yolo_root": Path(r"F:\Datasets\YOLO datasets\GLENDA_clean_yolo_seg"),
        "yaml_name": "glenda_clean_yolo_seg.yaml",
    },
    {
        "dataset_name": "GynSurg",
        "standardized_root": Path(
            r"C:\\Users\\cooll\\OneDrive\\Documents\\SPARC\\SplitDatasets\\GynSurg"
        ),
        "yolo_root": Path(r"C:\\Users\\cooll\\OneDrive\\Documents\\SPARC\\YOLO datasets\GynSurg_yolo_seg"),
        "yaml_name": "GynSurg_yolo_seg.yaml",
    },
]

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]

MIN_COMPONENT_AREA_PX = 3
CONTOUR_APPROX_EPSILON_FRACTION = 0.002


def find_matching_mask(masks_dir: Path, image_path: Path) -> Path:
    stem = image_path.stem

    for extension in IMAGE_EXTENSIONS:
        candidate = masks_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No mask found for image: {image_path}")


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    return (mask_np > 0).astype(np.uint8)

#Implemented to support multiclass contexts
def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    return mask_np


def contour_to_normalized_polygon(contour, width: int, height: int):
    contour = contour.reshape(-1, 2)

    if contour.shape[0] < 3:
        return None

    polygon = []

    for x, y in contour:
        x_norm = float(x) / float(width)
        y_norm = float(y) / float(height)

        x_norm = min(max(x_norm, 0.0), 1.0)
        y_norm = min(max(y_norm, 0.0), 1.0)

        polygon.extend([x_norm, y_norm])

    if len(polygon) < 6:
        return None

    return polygon


def fallback_box_polygon(component_mask: np.ndarray, width: int, height: int):
    ys, xs = np.where(component_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    if x2 <= x1 or y2 <= y1:
        return None

    points = [
        x1, y1,
        x2, y1,
        x2, y2,
        x1, y2,
    ]

    polygon = []

    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i + 1]

        polygon.extend(
            [
                min(max(float(x) / float(width), 0.0), 1.0),
                min(max(float(y) / float(height), 0.0), 1.0),
            ]
        )

    return polygon


def mask_to_yolo_segmentation_lines(mask_np: np.ndarray):
    height, width = mask_np.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_np.astype(np.uint8),
        connectivity=8,
    )

    label_lines = []
    component_count = 0

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < MIN_COMPONENT_AREA_PX:
            continue

        component_mask = (labels == label_id).astype(np.uint8)

        contours, _ = cv2.findContours(
            component_mask,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        arc_length = cv2.arcLength(largest_contour, closed=True)
        epsilon = CONTOUR_APPROX_EPSILON_FRACTION * arc_length

        approx = cv2.approxPolyDP(
            largest_contour,
            epsilon=epsilon,
            closed=True,
        )

        polygon = contour_to_normalized_polygon(
            contour=approx,
            width=width,
            height=height,
        )

        if polygon is None:
            polygon = fallback_box_polygon(
                component_mask=component_mask,
                width=width,
                height=height,
            )

        if polygon is None:
            continue

        values = ["0"] + [f"{value:.6f}" for value in polygon]
        label_lines.append(" ".join(values))
        component_count += 1

    return label_lines, component_count


def copy_image_to_yolo(image_path: Path, output_image_path: Path):
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output_image_path)


def write_label_file(label_lines, output_label_path: Path):
    output_label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_label_path, "w", encoding="utf-8") as file:
        if label_lines:
            file.write("\n".join(label_lines))
            file.write("\n")


def convert_split(dataset_name: str, standardized_root: Path, yolo_root: Path, split: str):
    images_dir = standardized_root / split / "images"
    masks_dir = standardized_root / split / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    output_images_dir = yolo_root / "images" / split
    output_labels_dir = yolo_root / "labels" / split

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for extension in IMAGE_EXTENSIONS:
        image_paths.extend(sorted(images_dir.glob(f"*{extension}")))

    rows = []

    for image_path in tqdm(image_paths, desc=f"{dataset_name} {split}"):
        mask_path = find_matching_mask(masks_dir, image_path)
        mask_np = load_binary_mask(mask_path)

        label_lines, component_count = mask_to_yolo_segmentation_lines(mask_np)

        output_image_path = output_images_dir / image_path.name
        output_label_path = output_labels_dir / f"{image_path.stem}.txt"

        copy_image_to_yolo(image_path, output_image_path)
        write_label_file(label_lines, output_label_path)

        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "image_name": image_path.name,
                "mask_name": mask_path.name,
                "num_instances": component_count,
                "has_foreground": int(mask_np.sum() > 0),
                "label_file": str(output_label_path),
            }
        )

    return rows


def write_dataset_yaml(yolo_root: Path, yaml_name: str):
    yaml_path = yolo_root / yaml_name

    data = {
        "path": str(yolo_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "lesion",
        },
    }

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    return yaml_path


def save_conversion_report(rows, yolo_root: Path, dataset_name: str):
    report_df = pd.DataFrame(rows)

    csv_path = yolo_root / f"{dataset_name}_yolo_seg_conversion_report.csv"
    xlsx_path = yolo_root / f"{dataset_name}_yolo_seg_conversion_report.xlsx"

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
                50,
            )

    print(f"Saved conversion CSV:  {csv_path}")
    print(f"Saved conversion XLSX: {xlsx_path}")


def convert_dataset(dataset_cfg):
    dataset_name = dataset_cfg["dataset_name"]
    standardized_root = dataset_cfg["standardized_root"]
    yolo_root = dataset_cfg["yolo_root"]
    yaml_name = dataset_cfg["yaml_name"]

    if yolo_root.exists():
        print(f"Removing existing YOLO dataset folder: {yolo_root}")
        shutil.rmtree(yolo_root)

    yolo_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    print("=" * 100)
    print(f"Converting {dataset_name} to YOLO segmentation format")
    print(f"Source: {standardized_root}")
    print(f"Target: {yolo_root}")
    print("=" * 100)

    for split in SPLITS:
        split_rows = convert_split(
            dataset_name=dataset_name,
            standardized_root=standardized_root,
            yolo_root=yolo_root,
            split=split,
        )

        all_rows.extend(split_rows)

    yaml_path = write_dataset_yaml(
        yolo_root=yolo_root,
        yaml_name=yaml_name,
    )

    save_conversion_report(
        rows=all_rows,
        yolo_root=yolo_root,
        dataset_name=dataset_name,
    )

    report_df = pd.DataFrame(all_rows)

    print("=" * 100)
    print(f"Finished {dataset_name}")
    print(f"YOLO YAML: {yaml_path}")
    print(report_df.groupby("split")["num_instances"].agg(["count", "sum", "mean"]))
    print("=" * 100)


def main():
    for dataset_cfg in DATASETS:
        convert_dataset(dataset_cfg)


if __name__ == "__main__":
    main()
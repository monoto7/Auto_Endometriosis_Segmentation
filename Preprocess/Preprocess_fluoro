from pathlib import Path
import shutil
import random
import re
from collections import defaultdict

import numpy as np
from PIL import Image


# ============================================================
# GLENDA binary pathology-region preparation script (all classes treated as one lesion class)
# ============================================================

# --- Input folders ---
FRAMES_DIR = Path(
    r"C:\Users\cooll\OneDrive\Documents\SPARC\Fluoro\fluoro-datagen-codex-playground\data\scene_2435673179\rgbs"
)

ANNOTS_DIR = Path(
    r"C:\Users\cooll\OneDrive\Documents\SPARC\Fluoro\fluoro-datagen-codex-playground\data\scene_2435673179\rgb_masks\vessel_tubes"
)

# --- Output folder ---
OUT_ROOT = Path(
    r"C:\Users\cooll\OneDrive\Documents\SPARC\SplitDatasets\Fluoro"
)

# --- Split settings ---
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

RANDOM_SEED = 42

# If True, deletes existing output folder before recreating it.
# Keep False unless you intentionally want to overwrite everything.
OVERWRITE_EXISTING = True


def extract_case_id(filename: str) -> str:
    """
    Extracts case ID from GLENDA-style filenames.

    Example:
      c_3_v_(video_29.mp4)_f_627.jpg -> c_3
      c_102_v_(video_3124.mp4)_f_97.jpg -> c_102

    If no case ID is found, the full stem is used as fallback.
    """
    match = re.match(r"^(c_\d+)_", filename)
    if match:
        return match.group(1)

    return Path(filename).stem


def collect_image_mask_pairs():
    """
    Collects all image-mask pairs.

    Expected:
      frames/xxx.jpg
      annots/xxx.png
    """
    image_paths = sorted(FRAMES_DIR.glob("*.png"))

    pairs = []
    missing_masks = []

    for image_path in image_paths:
        mask_path = ANNOTS_DIR / f"{image_path.stem}.png"

        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            missing_masks.append(image_path.name)

    print(f"Total image files found:       {len(image_paths)}")
    print(f"Image-mask pairs found:        {len(pairs)}")
    print(f"Images without matching mask:  {len(missing_masks)}")

    if missing_masks:
        print("\nFirst missing masks:")
        for name in missing_masks[:20]:
            print(f"  {name}")
        if len(missing_masks) > 20:
            print(f"  ... and {len(missing_masks) - 20} more")

    if not pairs:
        raise RuntimeError("No valid image-mask pairs found.")

    return pairs


def group_pairs_by_case(pairs):
    """
    Groups image-mask pairs by case ID to reduce leakage between train/val/test.
    """
    grouped = defaultdict(list)

    for image_path, mask_path in pairs:
        case_id = extract_case_id(image_path.name)
        grouped[case_id].append((image_path, mask_path))

    print(f"Unique case groups found:      {len(grouped)}")

    return grouped


def split_cases(grouped_pairs):
    """
    Splits case IDs into train/val/test.
    This keeps all frames from the same case in the same split.
    """
    case_ids = list(grouped_pairs.keys())

    random.seed(RANDOM_SEED)
    random.shuffle(case_ids)

    n_cases = len(case_ids)

    n_train = int(round(n_cases * TRAIN_RATIO))
    n_val = int(round(n_cases * VAL_RATIO))

    train_cases = case_ids[:n_train]
    val_cases = case_ids[n_train:n_train + n_val]
    test_cases = case_ids[n_train + n_val:]

    split_dict = {
        "train": [],
        "val": [],
        "test": [],
    }

    for case_id in train_cases:
        split_dict["train"].extend(grouped_pairs[case_id])

    for case_id in val_cases:
        split_dict["val"].extend(grouped_pairs[case_id])

    for case_id in test_cases:
        split_dict["test"].extend(grouped_pairs[case_id])

    print("\nCase-level split:")
    print(f"  Train cases: {len(train_cases)}")
    print(f"  Val cases:   {len(val_cases)}")
    print(f"  Test cases:  {len(test_cases)}")

    print("\nFrame-level split:")
    print(f"  Train frames: {len(split_dict['train'])}")
    print(f"  Val frames:   {len(split_dict['val'])}")
    print(f"  Test frames:  {len(split_dict['test'])}")

    return split_dict


def convert_glenda_mask_to_binary(mask_path: Path) -> Image.Image:
    """
    Converts GLENDA colored annotation masks to binary masks.

    GLENDA colors:
      background:       0 0 0
      Endo-Peritoneum:  190 62 204
      Endo-Ovar:        93 89 254
      Endo-TIE:         145 210 138
      Endo-Uterus:      238 236 50

    Output:
      background = 0
      pathology  = 255

    Since we want binary endometriosis/pathology vs background,
    all non-black pixels are treated as foreground.

    This handles both filled masks and box-like annotations, because any
    colored annotation pixel becomes white foreground.
    """
    mask = Image.open(mask_path).convert("RGB")
    mask_np = np.array(mask)

    binary = np.any(mask_np > 0, axis=-1).astype(np.uint8) * 255

    return Image.fromarray(binary, mode="L")


def prepare_output_dirs():
    """
    Creates output directory structure.
    """
    if OVERWRITE_EXISTING and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    for split_name in ["train", "val", "test"]:
        (OUT_ROOT / split_name / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split_name / "masks").mkdir(parents=True, exist_ok=True)


def process_split(split_name: str, pairs):
    """
    Copies images and saves binary masks for a split.
    """
    images_out = OUT_ROOT / split_name / "images"
    masks_out = OUT_ROOT / split_name / "masks"

    processed = 0
    failed = []

    for image_path, mask_path in pairs:
        try:
            image_dst = images_out / image_path.name
            mask_dst = masks_out / f"{image_path.stem}.png"

            # Copy original RGB laparoscopic image unchanged.
            shutil.copy2(image_path, image_dst)

            # Convert multi-class or box-like colored annotation to binary mask.
            binary_mask = convert_glenda_mask_to_binary(mask_path)
            binary_mask.save(mask_dst)

            processed += 1

        except Exception as e:
            failed.append((image_path.name, str(e)))

    print(f"\nProcessed {split_name}:")
    print(f"  Expected: {len(pairs)}")
    print(f"  Saved:    {processed}")
    print(f"  Failed:   {len(failed)}")

    if failed:
        print("\nFirst failures:")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")


def sanity_check_output():
    """
    Checks image-mask matching and prints foreground statistics.
    """
    print("\n" + "=" * 60)
    print("Final output sanity check")
    print("=" * 60)

    total_images = 0
    total_masks = 0

    for split_name in ["train", "val", "test"]:
        images_dir = OUT_ROOT / split_name / "images"
        masks_dir = OUT_ROOT / split_name / "masks"

        image_files = sorted(images_dir.glob("*.jpg"))
        mask_files = sorted(masks_dir.glob("*.png"))

        image_stems = {p.stem for p in image_files}
        mask_stems = {p.stem for p in mask_files}

        missing_masks = sorted(image_stems - mask_stems)
        missing_images = sorted(mask_stems - image_stems)

        foreground_masks = 0
        empty_masks = 0

        for mask_path in mask_files:
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)

            if np.any(mask_np > 0):
                foreground_masks += 1
            else:
                empty_masks += 1

        total_images += len(image_files)
        total_masks += len(mask_files)

        print(f"\n{split_name}")
        print(f"  images:              {len(image_files)}")
        print(f"  masks:               {len(mask_files)}")
        print(f"  masks with foreground: {foreground_masks}")
        print(f"  empty masks:           {empty_masks}")

        if missing_masks:
            print(f"  WARNING: images without masks: {len(missing_masks)}")
            for x in missing_masks[:10]:
                print(f"    {x}")

        if missing_images:
            print(f"  WARNING: masks without images: {len(missing_images)}")
            for x in missing_images[:10]:
                print(f"    {x}")

        if not missing_masks and not missing_images:
            print("  OK: image-mask stems match")

    print("\nTotal standardized dataset:")
    print(f"  images: {total_images}")
    print(f"  masks:  {total_masks}")


def check_split_leakage(split_dict):
    """
    Verifies that case IDs do not overlap across train/val/test.
    """
    split_cases = {}

    for split_name, pairs in split_dict.items():
        cases = set()
        for image_path, _ in pairs:
            cases.add(extract_case_id(image_path.name))
        split_cases[split_name] = cases

    train_val_overlap = split_cases["train"] & split_cases["val"]
    train_test_overlap = split_cases["train"] & split_cases["test"]
    val_test_overlap = split_cases["val"] & split_cases["test"]

    print("\nLeakage check:")
    print(f"  Train/val case overlap:  {len(train_val_overlap)}")
    print(f"  Train/test case overlap: {len(train_test_overlap)}")
    print(f"  Val/test case overlap:   {len(val_test_overlap)}")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("  WARNING: Case-level leakage detected.")
    else:
        print("  OK: no case-level overlap detected.")


def main():
    print("Preparing GLENDA 60/20/20 binary pathology-region dataset")
    print(f"Frames folder: {FRAMES_DIR}")
    print(f"Annots folder: {ANNOTS_DIR}")
    print(f"Output folder: {OUT_ROOT}")

    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames folder not found: {FRAMES_DIR}")

    if not ANNOTS_DIR.exists():
        raise FileNotFoundError(f"Annots folder not found: {ANNOTS_DIR}")

    prepare_output_dirs()

    pairs = collect_image_mask_pairs()
    grouped_pairs = group_pairs_by_case(pairs)
    split_dict = split_cases(grouped_pairs)

    check_split_leakage(split_dict)

    process_split("train", split_dict["train"])
    process_split("val", split_dict["val"])
    process_split("test", split_dict["test"])

    sanity_check_output()

    print("\nDone.")


if __name__ == "__main__":
    main()
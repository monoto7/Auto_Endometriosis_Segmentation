from pathlib import Path
import os
import shutil
import time
import inspect
import contextlib
import traceback

# =============================================================================
# Set nnU-Net paths BEFORE importing nnU-Net
# =============================================================================

NNUNET_BASE = Path(r"F:\Results\SAM_Benchmarking\nnUNet_1000epochs")

NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_BASE / "nnUNet_results"

os.environ["nnUNet_raw"] = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
os.environ["nnUNet_results"] = str(NNUNET_RESULTS)

# =============================================================================
# Imports after environment variables
# =============================================================================

import pandas as pd
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# =============================================================================
# User settings
# =============================================================================

OUTPUT_ROOT = NNUNET_BASE / "timed_test_inference_loaded_once"

DATASETS = {
    "ENID": {
        "dataset_id": "501",
        "dataset_folder": "Dataset501_ENID",
        "input_images": NNUNET_RAW / "Dataset501_ENID" / "imagesTs",
    },
    "GLENDA": {
        "dataset_id": "502",
        "dataset_folder": "Dataset502_GLENDA",
        "input_images": NNUNET_RAW / "Dataset502_GLENDA" / "imagesTs",
    },
    "GLENDA_clean": {
        "dataset_id": "503",
        "dataset_folder": "Dataset503_GLENDA_clean",
        "input_images": NNUNET_RAW / "Dataset503_GLENDA_clean" / "imagesTs",
    },
}

CONFIGURATION = "2d"
FOLD = "0"
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetPlans"

PREFERRED_CHECKPOINTS = [
    "checkpoint_best.pth",
    "checkpoint_final.pth",
    "checkpoint_latest.pth",
]

N_WARMUP = 3

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

SAVE_PROBABILITIES = False
OVERWRITE_OUTPUTS = True

# Use 1 process for cleaner and more stable per-case timing on Windows.
NUM_PROCESSES_PREPROCESSING = 1
NUM_PROCESSES_EXPORT = 1


# =============================================================================
# Path helpers
# =============================================================================

def get_model_folder(dataset_folder: str) -> Path:
    return (
        NNUNET_RESULTS
        / dataset_folder
        / f"{TRAINER}__{PLANS}__{CONFIGURATION}"
    )


def get_fold_folder(dataset_folder: str) -> Path:
    return get_model_folder(dataset_folder) / f"fold_{FOLD}"


def choose_checkpoint(fold_folder: Path) -> str:
    for checkpoint_name in PREFERRED_CHECKPOINTS:
        if (fold_folder / checkpoint_name).exists():
            return checkpoint_name

    available = sorted(p.name for p in fold_folder.glob("*.pth"))

    raise FileNotFoundError(
        f"No usable checkpoint found in:\n{fold_folder}\n\n"
        f"Expected one of:\n{PREFERRED_CHECKPOINTS}\n\n"
        f"Available checkpoints:\n{available}"
    )


def print_environment():
    print("\nUsing nnU-Net folders:")
    print(f"nnUNet_raw          = {os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed = {os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results      = {os.environ['nnUNet_results']}")
    print(f"Output root         = {OUTPUT_ROOT}")
    print(f"CUDA available      = {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU                 = {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint priority = {PREFERRED_CHECKPOINTS}\n")


def check_required_paths():
    required_paths = [
        NNUNET_RAW,
        NNUNET_PREPROCESSED,
        NNUNET_RESULTS,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    print("Checking dataset/model folders...")

    for dataset_name, info in DATASETS.items():
        dataset_folder = info["dataset_folder"]
        model_folder = get_model_folder(dataset_folder)
        fold_folder = get_fold_folder(dataset_folder)
        checkpoint_name = choose_checkpoint(fold_folder)

        required_dataset_paths = [
            info["input_images"],
            model_folder,
            model_folder / "dataset.json",
            model_folder / "plans.json",
            fold_folder,
            fold_folder / checkpoint_name,
        ]

        for path in required_dataset_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"[{dataset_name}] Missing required path: {path}"
                )

        print(f"[{dataset_name}] OK")
        print(f"  model folder: {model_folder}")
        print(f"  fold folder:  {fold_folder}")
        print(f"  checkpoint:   {checkpoint_name}")

    print("All required folders/files found.\n")


# =============================================================================
# Case handling
# =============================================================================

def get_image_files(images_folder: Path):
    files = []

    for ext in IMAGE_EXTENSIONS:
        files.extend(images_folder.glob(f"*{ext}"))

    return sorted(files)


def get_case_id_from_nnunet_image(image_path: Path) -> str:
    """
    nnU-Net image files are usually named like:
        ENID_test_000001_0000.png

    Case ID:
        ENID_test_000001
    """
    stem = image_path.stem

    if len(stem) >= 5 and stem[-5] == "_" and stem[-4:].isdigit():
        return stem[:-5]

    return stem


def get_cases(images_folder: Path):
    image_files = get_image_files(images_folder)

    cases = {}

    for image_file in image_files:
        case_id = get_case_id_from_nnunet_image(image_file)
        cases.setdefault(case_id, []).append(image_file)

    return dict(sorted(cases.items()))


def clear_folder(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def prepare_single_case_input(case_files, temp_input_dir: Path):
    clear_folder(temp_input_dir)

    for src_file in case_files:
        dst_file = temp_input_dir / src_file.name
        shutil.copy2(src_file, dst_file)


# =============================================================================
# Predictor initialization
# =============================================================================

def create_predictor():
    """
    Create nnU-Net predictor with version-compatible argument handling.
    Some nnU-Net v2 versions use perform_everything_on_device;
    older versions may use perform_everything_on_gpu.
    """
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    signature = inspect.signature(nnUNetPredictor.__init__)
    parameters = signature.parameters

    kwargs = {
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
        "device": device,
        "verbose": False,
        "verbose_preprocessing": False,
        "allow_tqdm": False,
    }

    if "perform_everything_on_device" in parameters:
        kwargs["perform_everything_on_device"] = True
    elif "perform_everything_on_gpu" in parameters:
        kwargs["perform_everything_on_gpu"] = torch.cuda.is_available()

    predictor = nnUNetPredictor(**kwargs)

    return predictor


def initialize_predictor_for_dataset(dataset_name: str, dataset_folder: str):
    model_folder = get_model_folder(dataset_folder)
    fold_folder = get_fold_folder(dataset_folder)
    checkpoint_name = choose_checkpoint(fold_folder)

    print(f"\nInitializing predictor for {dataset_name}")
    print(f"  model folder: {model_folder}")
    print(f"  fold:         {FOLD}")
    print(f"  checkpoint:   {checkpoint_name}")

    predictor = create_predictor()

    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(int(FOLD),),
        checkpoint_name=checkpoint_name,
    )

    return predictor, checkpoint_name


# =============================================================================
# Prediction / timing
# =============================================================================

def run_single_case_with_loaded_predictor(
    predictor,
    dataset_name: str,
    dataset_id: str,
    dataset_folder: str,
    checkpoint_name: str,
    case_id: str,
    case_files,
    case_index: int,
    total_cases: int,
    is_warmup: bool,
):
    dataset_output_root = OUTPUT_ROOT / dataset_name
    temp_input_dir = dataset_output_root / "_temp_single_input"

    if is_warmup:
        output_dir = dataset_output_root / "_warmup_outputs" / case_id
        log_file = dataset_output_root / "_logs" / f"warmup_{case_id}.txt"
    else:
        output_dir = dataset_output_root / "single_case_outputs" / case_id
        log_file = dataset_output_root / "_logs" / f"{case_id}.txt"

    clear_folder(output_dir)
    prepare_single_case_input(case_files, temp_input_dir)

    log_file.parent.mkdir(parents=True, exist_ok=True)

    label = "warm-up" if is_warmup else "timed"
    print(f"[{dataset_name}] {label} {case_index}/{total_cases}: {case_id}")

    start = None
    end = None
    status = "ok"
    error_message = ""

    try:
        with open(log_file, "w", encoding="utf-8", errors="ignore") as log:
            log.write(f"Dataset: {dataset_name}\n")
            log.write(f"Dataset ID: {dataset_id}\n")
            log.write(f"Dataset folder: {dataset_folder}\n")
            log.write(f"Case ID: {case_id}\n")
            log.write(f"Checkpoint: {checkpoint_name}\n")
            log.write(f"Input temp folder: {temp_input_dir}\n")
            log.write(f"Output folder: {output_dir}\n\n")

            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()

                predictor.predict_from_files(
                    str(temp_input_dir),
                    str(output_dir),
                    save_probabilities=SAVE_PROBABILITIES,
                    overwrite=OVERWRITE_OUTPUTS,
                    num_processes_preprocessing=NUM_PROCESSES_PREPROCESSING,
                    num_processes_segmentation_export=NUM_PROCESSES_EXPORT,
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0,
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end = time.perf_counter()

    except Exception as e:
        status = "failed"
        error_message = str(e)

        with open(log_file, "a", encoding="utf-8", errors="ignore") as log:
            log.write("\n\nERROR:\n")
            log.write(traceback.format_exc())

        print(f"  FAILED. See log: {log_file}")

        if start is not None and end is None:
            end = time.perf_counter()

    if start is None:
        elapsed_sec = None
    else:
        elapsed_sec = float(end - start)

    return {
        "dataset": dataset_name,
        "dataset_id": dataset_id,
        "dataset_folder": dataset_folder,
        "case_id": case_id,
        "input_files": "; ".join(str(p) for p in case_files),
        "n_input_files": len(case_files),
        "output_dir": str(output_dir),
        "log_file": str(log_file),
        "checkpoint_used": checkpoint_name,
        "inference_time_sec": elapsed_sec,
        "status": status,
        "error_message": error_message,
        "is_warmup": is_warmup,
        "timing_type": "loaded_once_per_dataset_preprocess_predict_postprocess_export",
    }


# =============================================================================
# Summary / output
# =============================================================================

def summarize(raw_df: pd.DataFrame):
    timed_ok = raw_df[
        (raw_df["is_warmup"] == False)
        & (raw_df["status"] == "ok")
    ].copy()

    if timed_ok.empty:
        return pd.DataFrame()

    summary = (
        timed_ok.groupby(["dataset", "checkpoint_used"], dropna=False)
        .agg(
            n_cases=("inference_time_sec", "count"),
            mean_time_sec=("inference_time_sec", "mean"),
            std_time_sec=("inference_time_sec", "std"),
            median_time_sec=("inference_time_sec", "median"),
            min_time_sec=("inference_time_sec", "min"),
            max_time_sec=("inference_time_sec", "max"),
        )
        .reset_index()
    )

    summary["mean_plus_minus_std_sec"] = summary.apply(
        lambda r: f"{r['mean_time_sec']:.4f} ± {r['std_time_sec']:.4f}",
        axis=1,
    )

    overall = pd.DataFrame(
        [
            {
                "dataset": "ALL",
                "checkpoint_used": "mixed_by_dataset",
                "n_cases": timed_ok["inference_time_sec"].count(),
                "mean_time_sec": timed_ok["inference_time_sec"].mean(),
                "std_time_sec": timed_ok["inference_time_sec"].std(),
                "median_time_sec": timed_ok["inference_time_sec"].median(),
                "min_time_sec": timed_ok["inference_time_sec"].min(),
                "max_time_sec": timed_ok["inference_time_sec"].max(),
                "mean_plus_minus_std_sec": (
                    f"{timed_ok['inference_time_sec'].mean():.4f} ± "
                    f"{timed_ok['inference_time_sec'].std():.4f}"
                ),
            }
        ]
    )

    summary = pd.concat([summary, overall], ignore_index=True)

    return summary


def save_outputs(raw_df: pd.DataFrame, summary_df: pd.DataFrame):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output_csv = OUTPUT_ROOT / "nnunetv2_1000epochs_test_inference_time_loaded_once_raw.csv"
    output_xlsx = OUTPUT_ROOT / "nnunetv2_1000epochs_test_inference_time_loaded_once_mean_std.xlsx"

    raw_df.to_csv(output_csv, index=False)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        raw_df.to_excel(writer, sheet_name="raw_times", index=False)

    print("\nSaved:")
    print(output_csv)
    print(output_xlsx)


# =============================================================================
# Main
# =============================================================================

def main():
    print_environment()
    check_required_paths()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_records = []

    for dataset_name, info in DATASETS.items():
        dataset_id = info["dataset_id"]
        dataset_folder = info["dataset_folder"]
        input_images = info["input_images"]

        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Dataset ID: {dataset_id}")
        print(f"Input folder: {input_images}")

        cases = get_cases(input_images)

        if len(cases) == 0:
            print(f"No nnU-Net image files found in: {input_images}")
            continue

        print(f"Cases found: {len(cases)}")

        case_items = list(cases.items())

        predictor, checkpoint_name = initialize_predictor_for_dataset(
            dataset_name=dataset_name,
            dataset_folder=dataset_folder,
        )

        # Warm-up
        n_warmup = min(N_WARMUP, len(case_items))
        print(f"Warm-up cases: {n_warmup}")

        warmup_failed = False

        for i, (case_id, case_files) in enumerate(case_items[:n_warmup], start=1):
            record = run_single_case_with_loaded_predictor(
                predictor=predictor,
                dataset_name=dataset_name,
                dataset_id=dataset_id,
                dataset_folder=dataset_folder,
                checkpoint_name=checkpoint_name,
                case_id=case_id,
                case_files=case_files,
                case_index=i,
                total_cases=n_warmup,
                is_warmup=True,
            )

            all_records.append(record)

            if record["status"] == "failed":
                warmup_failed = True
                break

        if warmup_failed:
            print(f"Skipping timed inference for {dataset_name} because warm-up failed.")
            del predictor

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        # Timed full test set
        print(f"Timed cases: {len(case_items)}")

        for i, (case_id, case_files) in enumerate(case_items, start=1):
            record = run_single_case_with_loaded_predictor(
                predictor=predictor,
                dataset_name=dataset_name,
                dataset_id=dataset_id,
                dataset_folder=dataset_folder,
                checkpoint_name=checkpoint_name,
                case_id=case_id,
                case_files=case_files,
                case_index=i,
                total_cases=len(case_items),
                is_warmup=False,
            )

            all_records.append(record)

        del predictor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_records) == 0:
        print("\nNo timing records generated.")
        return

    raw_df = pd.DataFrame(all_records)
    summary_df = summarize(raw_df)

    save_outputs(raw_df, summary_df)

    print("\nSummary:")
    if summary_df.empty:
        print("No successful timed cases.")
        print("Check the log files under:")
        print(OUTPUT_ROOT)
    else:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
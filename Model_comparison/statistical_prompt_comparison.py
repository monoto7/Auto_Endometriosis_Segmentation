from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(r"F:\Results\SAM_Benchmarking")

DATASETS = ["ENID", "GLENDA", "GLENDA_clean"]  # edit if folder names differ
SPLITS = ["test"]

MODELS = [
    "SAM2",
    "SurgiSAM2",
    "MedSAM",
    "SAM-Med2D",
]

PROMPTS = {
    "GT_point": "point",
    "GT_box": "box",
    "GT_box_point": "box_point",
    "GT_box_posneg": "box_posneg",
}

COMPARISONS = [
    ("box", "point"),
    ("box_point", "box"),
    ("box_posneg", "box"),
    ("box_posneg", "box_point"),
]

METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "f1",
    "tp_px",
    "fp_px",
    "fn_px",
]

OUTPUT_XLSX = ROOT / "simple_prompt_comparison_results.xlsx"


# ============================================================
# HELPERS
# ============================================================

def find_image_id_column(df):
    for col in ["image_name", "filename", "file_name", "image_id", "merged_mask_name", "gt_mask_name"]:
        if col in df.columns:
            return col
    raise ValueError("No image ID column found. Add your image filename column name to find_image_id_column().")


def read_all_results():
    rows = []

    for dataset in DATASETS:
        for split in SPLITS:
            for model in MODELS:
                for prompt_folder, prompt_name in PROMPTS.items():

                    path = ROOT / dataset / model / "frozen" / prompt_folder / split / "metrics_image_level.csv"

                    if not path.exists():
                        continue

                    df = pd.read_csv(path)

                    df["dataset"] = dataset
                    df["split"] = split
                    df["model"] = model
                    df["prompt"] = prompt_name
                    df["source_file"] = str(path)

                    rows.append(df)

    if not rows:
        raise RuntimeError("No CSV files found. Check ROOT, DATASETS, MODELS, and PROMPTS.")

    return pd.concat(rows, ignore_index=True)


def paired_prompt_test(df, dataset, split, metric, prompt_a, prompt_b):
    """
    Tests prompt_a vs prompt_b using paired image-model pairs.

    Example:
    prompt_a = box
    prompt_b = point

    Positive delta means box is better than point.
    """

    sub = df[
        (df["dataset"] == dataset) &
        (df["split"] == split) &
        (df["prompt"].isin([prompt_a, prompt_b]))
    ].copy()

    if sub.empty or metric not in sub.columns:
        return None

    id_col = find_image_id_column(sub)

    # Keep only models that support both prompt types
    models_a = set(sub[sub["prompt"] == prompt_a]["model"])
    models_b = set(sub[sub["prompt"] == prompt_b]["model"])
    compatible_models = sorted(models_a.intersection(models_b))

    if len(compatible_models) == 0:
        return None

    sub = sub[sub["model"].isin(compatible_models)]

    pivot = sub.pivot_table(
        index=["model", id_col],
        columns="prompt",
        values=metric,
        aggfunc="mean"
    ).reset_index()

    if prompt_a not in pivot.columns or prompt_b not in pivot.columns:
        return None

    paired = pivot.dropna(subset=[prompt_a, prompt_b]).copy()

    if len(paired) < 3:
        return None

    x = paired[prompt_a].astype(float)
    y = paired[prompt_b].astype(float)
    delta = x - y

    if np.allclose(delta, 0):
        stat = 0.0
        p_value = 1.0
    else:
        stat, p_value = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")

    return {
        "dataset": dataset,
        "split": split,
        "metric": metric,
        "comparison": f"{prompt_a} vs {prompt_b}",
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "compatible_models": ", ".join(compatible_models),
        "n_models": len(compatible_models),
        "n_image_model_pairs": len(paired),

        "mean_a": x.mean(),
        "mean_b": y.mean(),
        "mean_delta": delta.mean(),

        "median_a": x.median(),
        "median_b": y.median(),
        "median_delta": delta.median(),

        "std_delta": delta.std(ddof=1),
        "wilcoxon_stat": stat,
        "p_value": p_value,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

df = read_all_results()

available_metrics = [m for m in METRICS if m in df.columns]

print("Loaded rows:", len(df))
print("Datasets:", sorted(df["dataset"].unique()))
print("Models:", sorted(df["model"].unique()))
print("Prompts:", sorted(df["prompt"].unique()))
print("Available metrics:", available_metrics)

results = []

for dataset in DATASETS:
    for split in SPLITS:
        for metric in available_metrics:
            for prompt_a, prompt_b in COMPARISONS:
                res = paired_prompt_test(
                    df=df,
                    dataset=dataset,
                    split=split,
                    metric=metric,
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                )

                if res is not None:
                    results.append(res)

results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No valid prompt comparisons found.")


# ============================================================
# FDR correction within each dataset/split/metric only
# ============================================================

results_df["p_fdr"] = np.nan
results_df["significant_fdr_0.05"] = False

for (dataset, split, metric), group in results_df.groupby(["dataset", "split", "metric"]):
    idx = group.index
    pvals = group["p_value"].values

    reject, p_adj, _, _ = multipletests(
        pvals,
        alpha=0.05,
        method="fdr_bh"
    )

    results_df.loc[idx, "p_fdr"] = p_adj
    results_df.loc[idx, "significant_fdr_0.05"] = reject


# ============================================================
# Descriptive summary
# ============================================================

summary_rows = []

for (dataset, split, model, prompt), g in df.groupby(["dataset", "split", "model", "prompt"]):
    for metric in available_metrics:
        values = g[metric].dropna()

        if len(values) == 0:
            continue

        summary_rows.append({
            "dataset": dataset,
            "split": split,
            "model": model,
            "prompt": prompt,
            "metric": metric,
            "n_images": len(values),
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "median": values.median(),
            "q1": values.quantile(0.25),
            "q3": values.quantile(0.75),
        })

summary_df = pd.DataFrame(summary_rows)


# ============================================================
# Save
# ============================================================

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    results_df.to_excel(writer, index=False, sheet_name="prompt_tests")
    summary_df.to_excel(writer, index=False, sheet_name="summary")
    df.to_excel(writer, index=False, sheet_name="loaded_data")

print("\nSaved:")
print(OUTPUT_XLSX)
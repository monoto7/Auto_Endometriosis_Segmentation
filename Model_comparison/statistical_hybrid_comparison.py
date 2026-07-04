from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(r"F:\Results\SAM_Benchmarking")

OUTPUT_DIR = ROOT / "Model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_XLSX = OUTPUT_DIR / "hybrid_vs_segformer_statistics.xlsx"

DATASETS = ["ENID", "GLENDA_clean"]
SPLIT = "test"

METHOD_PATHS = {
    "SegFormer": Path("SegFormer") / "trained" / "No_prompt" / SPLIT / "metrics_image_level.csv",

    "Hybrid_AutoBox": Path("SegFormer_SurgiSAM2_AutoBox") / "hybrid" / "Auto_box" / SPLIT / "metrics_image_level.csv",

    "Hybrid_AutoBox_Fallback": Path("SegFormer_SurgiSAM2_AutoBox_Fallback") / "hybrid" / "Auto_box_fallback_dice_0p85_area_0p70_1p30" / SPLIT / "metrics_image_level.csv",

    "Hybrid_AutoBox_PosNeg_Fallback": Path("SegFormer_SurgiSAM2_AutoBox_PosNeg_Fallback") / "hybrid" / "Auto_box_posneg_fallback_dice_0p80" / SPLIT / "metrics_image_level.csv",
}

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

# Positive delta means method_a is better than method_b for metrics where higher is better.
# For fp_px and fn_px, lower is better, so interpret negative deltas as better.
COMPARISONS = [
    ("Hybrid_AutoBox", "SegFormer"),
    ("Hybrid_AutoBox_Fallback", "SegFormer"),
    ("Hybrid_AutoBox_PosNeg_Fallback", "SegFormer"),

    ("Hybrid_AutoBox_Fallback", "Hybrid_AutoBox"),
    ("Hybrid_AutoBox_PosNeg_Fallback", "Hybrid_AutoBox_Fallback"),
    ("Hybrid_AutoBox_PosNeg_Fallback", "Hybrid_AutoBox"),
]


# ============================================================
# HELPERS
# ============================================================

def find_image_id_column(df: pd.DataFrame) -> str:
    candidates = [
        "image_name",
        "filename",
        "file_name",
        "image_id",
        "case_id",
        "merged_mask_name",
        "gt_mask_name",
        "mask_name",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "No image ID column found. Add your image identifier column name "
        "to find_image_id_column(). Available columns are: "
        + ", ".join(df.columns)
    )


def load_all_image_level_results() -> pd.DataFrame:
    rows = []

    for dataset in DATASETS:
        for method, relative_path in METHOD_PATHS.items():
            path = ROOT / dataset / relative_path

            if not path.exists():
                print(f"[MISSING] {dataset} | {method}: {path}")
                continue

            df = pd.read_csv(path)
            df["dataset"] = dataset
            df["split"] = SPLIT
            df["method"] = method
            df["source_file"] = str(path)

            rows.append(df)

            print(f"[LOADED] {dataset} | {method}: {path}")

    if not rows:
        raise RuntimeError("No files were loaded. Check ROOT, DATASETS, and METHOD_PATHS.")

    return pd.concat(rows, ignore_index=True)


def summarize_methods(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []

    for (dataset, split, method), g in df.groupby(["dataset", "split", "method"], dropna=False):
        for metric in metrics:
            if metric not in g.columns:
                continue

            values = pd.to_numeric(g[metric], errors="coerce").dropna()

            if len(values) == 0:
                continue

            rows.append({
                "dataset": dataset,
                "split": split,
                "method": method,
                "metric": metric,
                "n_images": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
            })

    return pd.DataFrame(rows)


def paired_method_test(
    df: pd.DataFrame,
    dataset: str,
    split: str,
    metric: str,
    method_a: str,
    method_b: str,
) -> dict | None:
    sub = df[
        (df["dataset"] == dataset)
        & (df["split"] == split)
        & (df["method"].isin([method_a, method_b]))
    ].copy()

    if sub.empty:
        return None

    if metric not in sub.columns:
        return None

    methods_available = set(sub["method"].unique())
    if method_a not in methods_available or method_b not in methods_available:
        return None

    id_col = find_image_id_column(sub)

    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")

    pivot = sub.pivot_table(
        index=id_col,
        columns="method",
        values=metric,
        aggfunc="mean",
    ).reset_index()

    if method_a not in pivot.columns or method_b not in pivot.columns:
        return None

    paired = pivot.dropna(subset=[method_a, method_b]).copy()

    if len(paired) < 3:
        return {
            "dataset": dataset,
            "split": split,
            "metric": metric,
            "comparison": f"{method_a} vs {method_b}",
            "method_a": method_a,
            "method_b": method_b,
            "n_pairs": len(paired),
            "mean_a": np.nan,
            "mean_b": np.nan,
            "mean_delta": np.nan,
            "median_a": np.nan,
            "median_b": np.nan,
            "median_delta": np.nan,
            "std_delta": np.nan,
            "wilcoxon_stat": np.nan,
            "p_value": np.nan,
            "direction": "not_tested",
            "note": "Too few paired images",
        }

    x = paired[method_a].astype(float)
    y = paired[method_b].astype(float)
    delta = x - y

    if np.allclose(delta, 0, equal_nan=True):
        stat = 0.0
        p_value = 1.0
    else:
        stat, p_value = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
        )

    mean_delta = delta.mean()

    if metric in ["fp_px", "fn_px"]:
        if mean_delta < 0:
            direction = f"{method_a} lower/better than {method_b}"
        elif mean_delta > 0:
            direction = f"{method_a} higher/worse than {method_b}"
        else:
            direction = "no mean difference"
    else:
        if mean_delta > 0:
            direction = f"{method_a} higher/better than {method_b}"
        elif mean_delta < 0:
            direction = f"{method_a} lower/worse than {method_b}"
        else:
            direction = "no mean difference"

    return {
        "dataset": dataset,
        "split": split,
        "metric": metric,
        "comparison": f"{method_a} vs {method_b}",
        "method_a": method_a,
        "method_b": method_b,
        "n_pairs": len(paired),

        "mean_a": x.mean(),
        "mean_b": y.mean(),
        "mean_delta": mean_delta,

        "median_a": x.median(),
        "median_b": y.median(),
        "median_delta": delta.median(),

        "std_delta": delta.std(ddof=1),
        "delta_q1": delta.quantile(0.25),
        "delta_q3": delta.quantile(0.75),

        "wilcoxon_stat": stat,
        "p_value": p_value,
        "direction": direction,
        "note": "",
    }


def apply_fdr_correction(tests_df: pd.DataFrame) -> pd.DataFrame:
    tests_df = tests_df.copy()
    tests_df["p_fdr"] = np.nan
    tests_df["significant_fdr_0.05"] = False

    # Correction only within each dataset/split/metric family.
    for (dataset, split, metric), group in tests_df.groupby(["dataset", "split", "metric"], dropna=False):
        valid = group["p_value"].notna()

        if valid.sum() == 0:
            continue

        idx = group[valid].index
        pvals = tests_df.loc[idx, "p_value"].values

        reject, p_adj, _, _ = multipletests(
            pvals,
            alpha=0.05,
            method="fdr_bh",
        )

        tests_df.loc[idx, "p_fdr"] = p_adj
        tests_df.loc[idx, "significant_fdr_0.05"] = reject

    return tests_df


def get_best_methods(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, split, metric), g in summary_df.groupby(["dataset", "split", "metric"], dropna=False):
        if metric in ["fp_px", "fn_px"]:
            gg = g.sort_values("mean", ascending=True)
        else:
            gg = g.sort_values("mean", ascending=False)

        if len(gg) == 0:
            continue

        best = gg.iloc[0]

        rows.append({
            "dataset": dataset,
            "split": split,
            "metric": metric,
            "best_method_by_mean": best["method"],
            "best_mean": best["mean"],
            "best_median": best["median"],
            "n_images": best["n_images"],
            "note": "For fp_px/fn_px, lower is better. For other metrics, higher is better.",
        })

    return pd.DataFrame(rows)


def check_pairing(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in DATASETS:
        sub_dataset = df[df["dataset"] == dataset].copy()
        if sub_dataset.empty:
            continue

        id_col = find_image_id_column(sub_dataset)

        for method in METHOD_PATHS.keys():
            sub = sub_dataset[sub_dataset["method"] == method]
            if sub.empty:
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "n_unique_images": 0,
                    "image_id_column": id_col,
                    "note": "Method missing",
                })
            else:
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "n_unique_images": sub[id_col].nunique(),
                    "image_id_column": id_col,
                    "note": "",
                })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_all_image_level_results()

    available_metrics = [m for m in METRICS if m in df.columns]

    if not available_metrics:
        raise RuntimeError(
            "None of the requested metrics were found. Available columns are: "
            + ", ".join(df.columns)
        )

    print("\nLoaded rows:", len(df))
    print("Datasets:", sorted(df["dataset"].unique()))
    print("Methods:", sorted(df["method"].unique()))
    print("Available metrics:", available_metrics)

    pairing_check_df = check_pairing(df)
    summary_df = summarize_methods(df, available_metrics)

    test_rows = []

    for dataset in DATASETS:
        for metric in available_metrics:
            for method_a, method_b in COMPARISONS:
                res = paired_method_test(
                    df=df,
                    dataset=dataset,
                    split=SPLIT,
                    metric=metric,
                    method_a=method_a,
                    method_b=method_b,
                )

                if res is not None:
                    test_rows.append(res)

    tests_df = pd.DataFrame(test_rows)

    if tests_df.empty:
        raise RuntimeError("No valid paired comparisons were generated.")

    tests_df = apply_fdr_correction(tests_df)
    best_df = get_best_methods(summary_df)

    # Order sheets for readability.
    tests_df = tests_df.sort_values(
        by=["dataset", "metric", "comparison"],
        ascending=True,
    )

    summary_df = summary_df.sort_values(
        by=["dataset", "metric", "method"],
        ascending=True,
    )

    best_df = best_df.sort_values(
        by=["dataset", "metric"],
        ascending=True,
    )

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        tests_df.to_excel(writer, index=False, sheet_name="paired_tests")
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        best_df.to_excel(writer, index=False, sheet_name="best_method")
        pairing_check_df.to_excel(writer, index=False, sheet_name="pairing_check")
        df.to_excel(writer, index=False, sheet_name="loaded_image_level_data")

    print("\nSaved statistical analysis to:")
    print(OUTPUT_XLSX)


if __name__ == "__main__":
    main()
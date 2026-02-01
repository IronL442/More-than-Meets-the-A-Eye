import argparse
import os
from glob import glob
from typing import List

import pandas as pd


def _extract_fold(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    for part in parts:
        if part.startswith("fold_"):
            return part
    return "fold_unknown"


def _load_summaries(paths: List[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["fold"] = _extract_fold(path)
        frames.append(df)
    if not frames:
        raise ValueError("No non-empty summary CSVs found.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        type=str,
        default="outputs/eval/deepgaze_iie_finetuned_fold_*/ALL_SUMMARY.csv",
        help="Glob pattern for per-fold ALL_SUMMARY.csv files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/eval/deepgaze_iie_finetuned_cv_summary.csv",
        help="Output CSV path for averaged metrics.",
    )
    args = parser.parse_args()

    paths = sorted(glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {args.pattern}")

    df = _load_summaries(paths)
    group_cols = ["dataset", "split", "model"]
    metric_cols = [c for c in df.columns if c not in group_cols + ["fold"]]

    mean_df = (
        df.groupby(group_cols)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    std_df = (
        df.groupby(group_cols)[metric_cols]
        .std(numeric_only=True)
        .reset_index()
        .rename(columns={c: f"{c}_std" for c in metric_cols})
    )
    out_df = pd.merge(mean_df, std_df, on=group_cols, how="left")
    out_df.insert(0, "n_folds", df["fold"].nunique())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote averaged metrics to: {args.out}")


if __name__ == "__main__":
    main()

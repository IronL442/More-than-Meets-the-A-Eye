import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - optional dependency
    wilcoxon = None


LOWER_IS_BETTER = {"KL", "EMD", "KL_uniform", "EMD_uniform", "KL_centerbias", "EMD_centerbias"}
HIGHER_IS_BETTER = {"CC", "CC_uniform", "CC_centerbias"}


def _parse_list_arg(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    raise ValueError(
        f"Unknown metric direction for '{metric}'. "
        f"Add it to LOWER_IS_BETTER or HIGHER_IS_BETTER in this script."
    )


def _paired_wilcoxon(diff: np.ndarray) -> Dict[str, float]:
    if wilcoxon is None:
        return {"wilcoxon_stat": np.nan, "p_value": np.nan}
    nonzero = diff[np.abs(diff) > 0]
    if nonzero.size == 0:
        return {"wilcoxon_stat": 0.0, "p_value": 1.0}
    stat, p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", correction=False)
    return {"wilcoxon_stat": float(stat), "p_value": float(p)}


def compare(
    left_csv: str,
    right_csv: str,
    metrics: List[str],
    keys: List[str],
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    left = pd.read_csv(left_csv)
    right = pd.read_csv(right_csv)

    missing_keys = [k for k in keys if k not in left.columns or k not in right.columns]
    if missing_keys:
        raise KeyError(f"Missing key columns in CSVs: {missing_keys}")

    left_cols = keys + [m for m in metrics if m in left.columns]
    right_cols = keys + [m for m in metrics if m in right.columns]
    merged = pd.merge(
        left[left_cols],
        right[right_cols],
        on=keys,
        suffixes=(f"_{left_label}", f"_{right_label}"),
        how="inner",
    )

    rows = []
    for metric in metrics:
        c_left = f"{metric}_{left_label}"
        c_right = f"{metric}_{right_label}"
        if c_left not in merged.columns or c_right not in merged.columns:
            continue

        pair = merged[[c_left, c_right]].dropna()
        if pair.empty:
            continue

        l = pair[c_left].to_numpy(dtype=np.float64)
        r = pair[c_right].to_numpy(dtype=np.float64)
        diff = r - l  # right - left
        direction = _metric_direction(metric)

        if direction == "lower":
            improved = r < l
            improvement_signed = l - r  # positive == right better
        else:
            improved = r > l
            improvement_signed = r - l  # positive == right better

        wil = _paired_wilcoxon(diff)
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "n_pairs": int(len(pair)),
                f"mean_{left_label}": float(l.mean()),
                f"mean_{right_label}": float(r.mean()),
                "mean_delta_right_minus_left": float(diff.mean()),
                "mean_improvement_right_vs_left": float(improvement_signed.mean()),
                "median_improvement_right_vs_left": float(np.median(improvement_signed)),
                "pct_images_improved_right_vs_left": float(100.0 * improved.mean()),
                "wilcoxon_stat": wil["wilcoxon_stat"],
                "p_value": wil["p_value"],
            }
        )

    if not rows:
        raise ValueError("No comparable metric columns found after merge.")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired metric comparison with Wilcoxon test.")
    ap.add_argument("--left_csv", required=True, help="CSV for reference model (e.g., baseline).")
    ap.add_argument("--right_csv", required=True, help="CSV for comparison model (e.g., finetuned).")
    ap.add_argument("--left_label", default="base", help="Short label suffix for left model.")
    ap.add_argument("--right_label", default="ft", help="Short label suffix for right model.")
    ap.add_argument(
        "--metrics",
        default="KL,CC,EMD",
        help="Comma-separated metrics to compare. Default: KL,CC,EMD",
    )
    ap.add_argument(
        "--keys",
        default="dataset,split,image_id",
        help="Comma-separated merge keys for pairing rows. Default: dataset,split,image_id",
    )
    ap.add_argument(
        "--out",
        default="outputs/eval/model_comparison_stats.csv",
        help="Output CSV path.",
    )
    args = ap.parse_args()

    metrics = _parse_list_arg(args.metrics)
    keys = _parse_list_arg(args.keys)
    if not metrics:
        raise ValueError("At least one metric is required.")
    if not keys:
        raise ValueError("At least one key is required.")

    out_df = compare(
        left_csv=args.left_csv,
        right_csv=args.right_csv,
        metrics=metrics,
        keys=keys,
        left_label=args.left_label,
        right_label=args.right_label,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(out_df.to_string(index=False))
    print(f"\nWrote comparison stats to: {args.out}")
    if wilcoxon is None:
        print("Note: scipy is not installed; Wilcoxon stats are NaN.")


if __name__ == "__main__":
    main()

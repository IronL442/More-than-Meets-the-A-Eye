import argparse
import os
import re
from glob import glob
from typing import Dict, List, Tuple

# Keep matplotlib/font cache writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SOURCES = {
    "Baseline": "outputs/eval/baseline_seminar/ALL_SUMMARY.csv",
    "Finetuned": "outputs/non_augmented/eval/deepgaze_iie_finetuned_fold_*/ALL_SUMMARY.csv",
    "MiaMix": "outputs/miamix/eval/finetuned_fold_*/ALL_SUMMARY.csv",
    "AugSal": "outputs/augsal/eval/finetuned_fold_*/ALL_SUMMARY.csv",
    "AugSalStrong": "outputs/augsal_strong/eval/finetuned_fold_*/ALL_SUMMARY.csv",
}

METRIC_DIRECTIONS = {
    "CC": "higher",
    "KL": "lower",
    "EMD": "lower",
    "CC_uniform": "higher",
    "KL_uniform": "lower",
    "EMD_uniform": "lower",
}


def _df_to_markdown(df: pd.DataFrame) -> str:
    headers = df.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _extract_run_id(path: str) -> str:
    match = re.search(r"(fold_\d+)", os.path.normpath(path))
    if match:
        return match.group(1)
    return os.path.basename(os.path.dirname(path)) or "single_run"


def _resolve_paths(pattern_or_path: str) -> List[str]:
    if any(ch in pattern_or_path for ch in ["*", "?", "["]):
        return sorted(glob(pattern_or_path))
    return [pattern_or_path] if os.path.exists(pattern_or_path) else []


def _load_method_rows(method: str, pattern_or_path: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in _resolve_paths(pattern_or_path):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["method"] = method
        df["run_id"] = _extract_run_id(path)
        df["source_path"] = path
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No result CSVs found for {method}: {pattern_or_path}")
    return pd.concat(rows, ignore_index=True)


def _metric_columns(df: pd.DataFrame) -> List[str]:
    protected = {"dataset", "split", "model", "method", "run_id", "source_path"}
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    preferred = [m for m in ["CC", "KL", "EMD"] if m in numeric]
    extras = [m for m in ["CC_uniform", "KL_uniform", "EMD_uniform"] if m in numeric]
    return preferred + extras + sorted([c for c in numeric if c not in set(preferred + extras) and c not in protected])


def _aggregate(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    gcols = ["method", "dataset", "split", "model"]
    agg_spec = {"run_id": "nunique"}
    agg_spec.update({m: ["mean", "std"] for m in metrics})
    out = df.groupby(gcols, dropna=False).agg(agg_spec).reset_index()

    flat_cols: List[str] = []
    for c in out.columns:
        if isinstance(c, tuple):
            if c[1] == "":
                flat_cols.append(c[0])
            elif c[0] == "run_id":
                flat_cols.append("n_runs")
            else:
                flat_cols.append(f"{c[0]}_{c[1]}")
        else:
            flat_cols.append(c)
    out.columns = flat_cols
    for m in metrics:
        std_col = f"{m}_std"
        if std_col in out.columns:
            out[std_col] = out[std_col].fillna(0.0)
    return out


def _attach_baseline_deltas(agg: pd.DataFrame, metrics: List[str], baseline_label: str = "Baseline") -> pd.DataFrame:
    keys = ["dataset", "split", "model"]
    base_cols = keys + [f"{m}_mean" for m in metrics]
    base = agg[agg["method"] == baseline_label][base_cols].copy()
    base = base.rename(columns={f"{m}_mean": f"{m}_baseline" for m in metrics})

    merged = agg.merge(base, on=keys, how="left")
    for m in metrics:
        bcol = f"{m}_baseline"
        dcol = f"{m}_delta_vs_baseline"
        pcol = f"{m}_pct_vs_baseline"
        mean_col = f"{m}_mean"
        direction = METRIC_DIRECTIONS.get(m, "higher")

        if direction == "lower":
            merged[dcol] = merged[bcol] - merged[mean_col]
        else:
            merged[dcol] = merged[mean_col] - merged[bcol]

        # Positive percentage means better than baseline according to metric direction.
        denom = merged[bcol].abs().replace(0, np.nan)
        merged[pcol] = (100.0 * merged[dcol] / denom).replace([np.inf, -np.inf], np.nan)
    return merged


def _order_methods(values: pd.Series, direction: str) -> List[str]:
    ascending = direction == "lower"
    return values.sort_values(ascending=ascending).index.tolist()


def _plot_metric_bars(agg: pd.DataFrame, metrics: List[str], out_path: str) -> None:
    plot_df = agg.copy()
    methods = sorted(plot_df["method"].unique().tolist())
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    palette = {
        "Baseline": "#5B8FF9",
        "Finetuned": "#61DDAA",
        "MiaMix": "#65789B",
        "AugSal": "#F6BD16",
        "AugSalStrong": "#F08BB4",
    }

    for ax, metric in zip(axes, metrics):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        data = plot_df.set_index("method")[[mean_col, std_col]]
        direction = METRIC_DIRECTIONS.get(metric, "higher")
        ordered = _order_methods(data[mean_col], direction)
        means = data.loc[ordered, mean_col].to_numpy(dtype=float)
        stds = data.loc[ordered, std_col].to_numpy(dtype=float)
        x = np.arange(len(ordered))
        colors = [palette.get(m, "#8D8D8D") for m in ordered]

        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="#222222", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=30, ha="right")
        ax.set_title(f"{metric} ({'higher better' if direction == 'higher' else 'lower better'})")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Seminar Evaluation Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_heatmap(agg_with_delta: pd.DataFrame, metrics: List[str], out_path: str) -> None:
    keep_cols = ["method"] + [f"{m}_delta_vs_baseline" for m in metrics]
    hdf = agg_with_delta[keep_cols].copy().set_index("method")
    hdf = hdf.loc[[m for m in ["Baseline", "Finetuned", "MiaMix", "AugSal", "AugSalStrong"] if m in hdf.index]]
    mat = hdf.to_numpy(dtype=float)

    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(1.6 * len(metrics) + 2.5, 0.8 * len(hdf) + 2.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(hdf.index)))
    ax.set_yticklabels(hdf.index.tolist())
    ax.set_title("Delta vs Baseline (positive is better)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = "nan" if not np.isfinite(val) else f"{val:.4f}"
            ax.text(j, i, txt, ha="center", va="center", color="#1f1f1f", fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Improvement")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _normalize_for_radar(agg: pd.DataFrame, metrics: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    rdf = agg[["method"] + [f"{m}_mean" for m in metrics]].copy().set_index("method")
    out = pd.DataFrame(index=rdf.index)
    for m in metrics:
        col = f"{m}_mean"
        vals = rdf[col].to_numpy(dtype=float)
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin == 0:
            scores = np.ones_like(vals)
        else:
            if METRIC_DIRECTIONS.get(m, "higher") == "lower":
                scores = (vmax - vals) / (vmax - vmin)
            else:
                scores = (vals - vmin) / (vmax - vmin)
        out[m] = scores
    return out, metrics


def _plot_radar(agg: pd.DataFrame, metrics: List[str], out_path: str) -> None:
    radar_df, labels = _normalize_for_radar(agg, metrics)
    methods = [m for m in ["Baseline", "Finetuned", "MiaMix", "AugSal", "AugSalStrong"] if m in radar_df.index]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"polar": True})
    palette = {
        "Baseline": "#5B8FF9",
        "Finetuned": "#61DDAA",
        "MiaMix": "#65789B",
        "AugSal": "#F6BD16",
        "AugSalStrong": "#F08BB4",
    }

    for method in methods:
        vals = radar_df.loc[method, labels].to_numpy(dtype=float).tolist()
        vals += vals[:1]
        color = palette.get(method, "#7F7F7F")
        ax.plot(angles, vals, linewidth=2.0, label=method, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Normalized Multi-Metric Score (higher is better)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_overview_markdown(agg_with_delta: pd.DataFrame, metrics: List[str], out_path: str) -> None:
    cols = ["method", "n_runs"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]
    core = agg_with_delta[cols].copy()
    for m in metrics:
        core[f"{m}_mean"] = core[f"{m}_mean"].map(lambda x: f"{x:.6f}")
        core[f"{m}_std"] = core[f"{m}_std"].map(lambda x: f"{x:.6f}")

    lines = [
        "# Seminar Results Overview",
        "",
        "## Mean ± Std Across Runs/Folds",
        "",
        _df_to_markdown(core),
        "",
        "## Delta vs Baseline (positive is better)",
        "",
    ]
    dcols = ["method"] + [f"{m}_delta_vs_baseline" for m in metrics] + [f"{m}_pct_vs_baseline" for m in metrics]
    delta = agg_with_delta[dcols].copy()
    for c in delta.columns:
        if c == "method":
            continue
        delta[c] = delta[c].map(lambda x: "nan" if pd.isna(x) else f"{x:.6f}")
    lines.append(_df_to_markdown(delta))
    lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Create consolidated summary tables and visualizations for seminar experiments.")
    ap.add_argument("--out_dir", default="outputs/overview", help="Output directory for overview artifacts.")
    ap.add_argument("--baseline", default=DEFAULT_SOURCES["Baseline"], help="Baseline ALL_SUMMARY.csv path.")
    ap.add_argument("--finetuned", default=DEFAULT_SOURCES["Finetuned"], help="Glob for finetuned ALL_SUMMARY.csv files.")
    ap.add_argument("--miamix", default=DEFAULT_SOURCES["MiaMix"], help="Glob for MiaMix ALL_SUMMARY.csv files.")
    ap.add_argument("--augsal", default=DEFAULT_SOURCES["AugSal"], help="Glob for AugSal ALL_SUMMARY.csv files.")
    ap.add_argument("--augsal_strong", default=DEFAULT_SOURCES["AugSalStrong"], help="Glob for strong AugSal ALL_SUMMARY.csv files.")
    ap.add_argument(
        "--plot_metrics",
        default="CC,KL,EMD",
        help="Comma-separated metrics used for plots/ranking. Defaults to CC,KL,EMD.",
    )
    args = ap.parse_args()

    sources = {
        "Baseline": args.baseline,
        "Finetuned": args.finetuned,
        "MiaMix": args.miamix,
        "AugSal": args.augsal,
        "AugSalStrong": args.augsal_strong,
    }

    frames = []
    for method, src in sources.items():
        frames.append(_load_method_rows(method, src))
    raw = pd.concat(frames, ignore_index=True)

    metrics = _metric_columns(raw)
    if not metrics:
        raise ValueError("No numeric metric columns found in inputs.")

    agg = _aggregate(raw, metrics)
    agg_with_delta = _attach_baseline_deltas(agg, metrics, baseline_label="Baseline")

    requested_plot_metrics = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
    plot_metrics = [m for m in requested_plot_metrics if m in metrics]
    if not plot_metrics:
        raise ValueError(f"None of the requested plot metrics were found: {requested_plot_metrics}")

    os.makedirs(args.out_dir, exist_ok=True)

    raw_out = os.path.join(args.out_dir, "seminar_raw_runs.csv")
    agg_out = os.path.join(args.out_dir, "seminar_overview_metrics.csv")
    md_out = os.path.join(args.out_dir, "seminar_overview.md")
    bars_out = os.path.join(args.out_dir, "seminar_metric_bars.png")
    heat_out = os.path.join(args.out_dir, "seminar_delta_heatmap.png")
    radar_out = os.path.join(args.out_dir, "seminar_radar.png")

    raw.to_csv(raw_out, index=False)
    agg_with_delta.to_csv(agg_out, index=False)
    _build_overview_markdown(agg_with_delta, metrics=plot_metrics, out_path=md_out)
    _plot_metric_bars(agg_with_delta, metrics=plot_metrics, out_path=bars_out)
    _plot_delta_heatmap(agg_with_delta, metrics=plot_metrics, out_path=heat_out)
    _plot_radar(agg_with_delta, metrics=plot_metrics, out_path=radar_out)

    print(f"Wrote: {raw_out}")
    print(f"Wrote: {agg_out}")
    print(f"Wrote: {md_out}")
    print(f"Wrote: {bars_out}")
    print(f"Wrote: {heat_out}")
    print(f"Wrote: {radar_out}")


if __name__ == "__main__":
    main()

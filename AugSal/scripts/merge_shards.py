from __future__ import annotations

import argparse
import csv
import json
import shutil
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file()])


def _copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def _copy_tree(src_root: Path, dst_root: Path, mode: str) -> int:
    count = 0
    for src in _iter_files(src_root):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        _copy_file(src, dst, mode=mode)
        count += 1
    return count


def _read_metadata_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _read_run_summary(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = str(row.get("image_id", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    out.sort(key=lambda r: str(r.get("image_id", "")))
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metadata rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge AugSal shard outputs into one dataset root.")
    parser.add_argument(
        "--shards_root",
        type=str,
        default="/kaggle/working/AugSal/shards",
        help="Directory containing shard_* output folders.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="shard_*",
        help="Glob pattern under shards_root for shard folders.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/kaggle/working/AugSal/augmented_data",
        help="Merged output root.",
    )
    parser.add_argument(
        "--copy_mode",
        type=str,
        default="copy",
        choices=["copy", "hardlink"],
        help="How to materialize merged files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing out_root before merge.",
    )
    args = parser.parse_args()

    shards_root = Path(args.shards_root)
    out_root = Path(args.out_root)

    shard_paths = sorted([Path(p) for p in glob(str(shards_root / args.pattern)) if Path(p).is_dir()])
    if not shard_paths:
        raise FileNotFoundError(f"No shard folders found under {shards_root} with pattern {args.pattern}")

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    subdirs = ["images", "gt_maps", "change_maps", "selected_attention_maps"]
    copied: Dict[str, int] = {k: 0 for k in subdirs}

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for shard in shard_paths:
        for sub in subdirs:
            src = shard / sub
            dst = out_root / sub
            copied[sub] += _copy_tree(src, dst, mode=args.copy_mode)

        all_rows.extend(_read_metadata_csv(shard / "metadata.csv"))
        summary = _read_run_summary(shard / "run_summary.json")
        if summary is not None:
            summary["shard_path"] = str(shard)
            summaries.append(summary)

    merged_rows = _dedupe_rows(all_rows)
    _write_csv(out_root / "metadata.csv", merged_rows)
    _write_jsonl(out_root / "metadata.jsonl", merged_rows)

    merge_summary = {
        "num_shards": len(shard_paths),
        "shards": [str(s) for s in shard_paths],
        "copy_mode": args.copy_mode,
        "copied_files": copied,
        "metadata_rows_in": len(all_rows),
        "metadata_rows_out": len(merged_rows),
        "run_summaries": summaries,
    }
    with open(out_root / "merge_summary.json", "w", encoding="utf-8") as f:
        json.dump(merge_summary, f, indent=2)

    print(json.dumps(merge_summary, indent=2))


if __name__ == "__main__":
    main()

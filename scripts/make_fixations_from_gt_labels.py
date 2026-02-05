from __future__ import annotations

"""
Convert per-participant GT density PNGs in data/seminar_data/gt_labels
into binary fixation masks expected by FolderDataset.

Output: data/seminar_data/fixations/<stem>.npy (uint8, 0/1 mask).
The mask is the union of all participant maps for a given image.

Usage (from repo root, venv active):
    python scripts/make_fixations_from_gt_labels.py
"""

import os
import re
import pathlib
import cv2
import numpy as np


def main() -> None:
    root = pathlib.Path("data/seminar_data/gt_labels")
    out_dir = pathlib.Path("data/seminar_data/fixations")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Files are named like P01_IMG001_10100.png
    # Example filename: P01_IMG001_10100.png -> stem = IMG001_10100
    pat = re.compile(r"P\d+_(IMG\d+_\d+)\.png", re.IGNORECASE)

    groups: dict[str, list[pathlib.Path]] = {}
    for p in root.glob("P*.png"):
        m = pat.match(p.name)
        if not m:
            continue
        stem = m.group(1)
        groups.setdefault(stem, []).append(p)

    if not groups:
        raise FileNotFoundError(f"No matching PNGs in {root}")

    for stem, paths in groups.items():
        # Target resolution comes from the source image
        img_path = pathlib.Path("data/seminar_data/images") / f"{stem}.jpg"
        if not img_path.exists():
            img_path = img_path.with_suffix(".png")
        if not img_path.exists():
            raise FileNotFoundError(f"Source image not found for {stem} (looked for .jpg/.png)")
        img_ref = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_ref is None:
            raise FileNotFoundError(f"Failed to read source image: {img_path}")
        H, W = img_ref.shape[:2]

        acc = None
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to read {p}")
            if img.ndim == 3:
                if img.shape[2] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            if gray.shape != (H, W):
                gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_NEAREST)

            mask = (gray > 0).astype(np.uint8)
            acc = mask if acc is None else acc + mask

        fix = (acc > 0).astype(np.uint8)  # union of all participants
        np.save(out_dir / f"{stem}.npy", fix)

    print(f"Wrote fixation masks to {out_dir}")


if __name__ == "__main__":
    main()

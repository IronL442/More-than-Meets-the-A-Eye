import argparse
import os
from typing import List

import numpy as np


def _load_image_paths(root: str) -> List[str]:
    import glob
    img_dir = os.path.join(root, "images")
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(sorted([p for p in glob.glob(os.path.join(img_dir, ext))]))
    return paths


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _write_list(path: str, items: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root with images/ folder")
    parser.add_argument("--test_count", type=int, default=30, help="Number of images to hold out")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out_dir", type=str, default="splits")
    args = parser.parse_args()

    paths = _load_image_paths(args.root)
    if not paths:
        raise FileNotFoundError(f"No images found under {args.root}/images")

    n = len(paths)
    if args.test_count <= 0 or args.test_count >= n:
        raise ValueError("test_count must be > 0 and < number of images.")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=args.test_count, replace=False)
    test_paths = [paths[i] for i in idx]
    test_ids = sorted({_stem(p) for p in test_paths})
    trainval_ids = sorted({_stem(p) for p in paths if _stem(p) not in set(test_ids)})

    os.makedirs(args.out_dir, exist_ok=True)
    test_path = os.path.join(args.out_dir, "test.txt")
    trainval_path = os.path.join(args.out_dir, "trainval.txt")
    _write_list(test_path, test_ids)
    _write_list(trainval_path, trainval_ids)

    print(f"Total images: {n}")
    print(f"Test images: {len(test_ids)} -> {test_path}")
    print(f"Train/val images: {len(trainval_ids)} -> {trainval_path}")


if __name__ == "__main__":
    main()

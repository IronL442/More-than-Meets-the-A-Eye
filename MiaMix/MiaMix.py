from __future__ import annotations

import argparse
import csv
import json
import os
import random
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image

Tensor = torch.Tensor


def to_prob_map(arr: np.ndarray) -> np.ndarray:
    """Convert a non-negative saliency map to a normalized probability map."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total > 0.0:
        return arr / total
    h, w = arr.shape
    return np.full((h, w), 1.0 / float(h * w), dtype=np.float32)


def sample_dirichlet(alpha: float, k: int) -> np.ndarray:
    """Sample lam_1..lam_(k+1) from Dir(alpha,...,alpha)."""
    alpha_vec = np.full(k + 1, alpha, dtype=np.float32)
    return np.random.dirichlet(alpha_vec)


class MiAMix(nn.Module):
    """
    Multi-stage Augmented Mixup (MiaMix).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        k_max: int = 3,
        prob_self: float = 0.1,
        methods: List[str] = ("mixup", "cutmix", "agmix"),
        method_weights: List[float] | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.k_max = int(k_max)
        self.prob_self = float(prob_self)
        self.methods = list(methods)
        self.device = device

        if method_weights is None:
            self.method_weights = np.ones(len(self.methods), dtype=np.float64) / float(len(self.methods))
        else:
            if len(method_weights) != len(self.methods):
                raise ValueError("method_weights length must match methods length.")
            w = np.asarray(method_weights, dtype=np.float64)
            if np.any(w < 0.0) or float(w.sum()) <= 0.0:
                raise ValueError("method_weights must be non-negative with positive sum.")
            self.method_weights = w / float(w.sum())

    def augment_mask(self, mask: Tensor) -> Tensor:
        angle = float(np.random.uniform(-15.0, 15.0))
        mask_b = mask.unsqueeze(0)
        theta = torch.tensor(
            [[
                [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0.0],
                [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.0],
            ]],
            device=self.device,
            dtype=torch.float32,
        )
        grid = F.affine_grid(theta, mask_b.size(), align_corners=False)
        mask = F.grid_sample(mask_b, grid, align_corners=False).squeeze(0)
        k = int(np.random.choice([3, 5, 7]))
        sigma = float(np.random.uniform(0.3, 1.2))
        mask = TF.gaussian_blur(mask, [k, k], [sigma, sigma])
        return mask.clamp(0.0, 1.0)

    def cutmix_mask(self, h: int, w: int, lam: float) -> Tensor:
        cut_w = int(w * np.sqrt(1.0 - lam))
        cut_h = int(h * np.sqrt(1.0 - lam))
        cx, cy = int(np.random.randint(w)), int(np.random.randint(h))

        mask = torch.ones(1, h, w, device=self.device)
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, h)
        mask[:, y1:y2, x1:x2] = 0.0
        return mask

    def agmix_mask(self, h: int, w: int, lam: float) -> Tensor:
        mu_x, mu_y = float(np.random.uniform(0, w)), float(np.random.uniform(0, h))
        sigma = np.sqrt(max(1e-8, 1.0 - lam)) * max(h, w) / 4.0
        y, x = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing="ij",
        )
        g = torch.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2.0 * sigma**2))
        return (g > 0.5).float().unsqueeze(0)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, List[bool]]:
        n, _, h, w = x.shape
        out_x, out_y = [], []
        self_mix_flags: List[bool] = []

        for i in range(n):
            if float(np.random.rand()) < self.prob_self:
                j = i
                self_mix = True
            else:
                j = int(np.random.randint(0, n))
                self_mix = j == i

            x_i, y_i = x[i], y[i]
            x_t, y_t = x[j], y[j]

            k = int(np.random.randint(1, self.k_max + 1))
            lambdas = sample_dirichlet(self.alpha, k)
            methods = np.random.choice(self.methods, size=k, p=self.method_weights)

            masks = []
            for j_stage in range(k):
                lam_j = float(lambdas[j_stage])
                method = str(methods[j_stage])
                if method == "mixup":
                    mask = torch.full((1, h, w), lam_j, device=self.device)
                elif method == "cutmix":
                    mask = self.cutmix_mask(h, w, lam_j)
                elif method == "agmix":
                    mask = self.agmix_mask(h, w, lam_j)
                else:
                    raise ValueError(f"Unknown mix method: {method}")
                mask = self.augment_mask(mask)
                masks.append(mask)

            mask_merged = torch.stack(masks).mean(dim=0)
            lambda_merged = float(mask_merged.mean().item())
            mask_x = mask_merged.repeat(x_i.shape[0], 1, 1)

            x_mix = mask_x * x_i + (1.0 - mask_x) * x_t
            y_mix = lambda_merged * y_i + (1.0 - lambda_merged) * y_t

            out_x.append(x_mix)
            out_y.append(y_mix)
            self_mix_flags.append(self_mix)

        return torch.stack(out_x), torch.stack(out_y), self_mix_flags


def _stem(path: str) -> str:
    return Path(path).stem


def _load_id_list(path: str | None) -> set[str]:
    ids: set[str] = set()
    if not path:
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            ids.add(_stem(item))
    return ids


def _list_images(image_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob(os.path.join(image_dir, ext)))
    return sorted(paths)


def _set_reproducibility(
    *,
    seed: int,
    deterministic: bool,
    warn_only: bool,
    cublas_workspace_config: str | None,
) -> dict:
    if cublas_workspace_config:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(cublas_workspace_config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)

    torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))

    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "warn_only": bool(warn_only),
        "cublas_workspace_config": cublas_workspace_config,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def _parse_csv_list(raw: str, cast=float) -> List:
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _resolve_device(requested: str) -> str:
    req = requested.lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return req


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MiaMix-augmented saliency training pairs.")
    parser.add_argument("--image_dir", default="data/seminar_data/images")
    parser.add_argument("--heatmap_dir", default="data/seminar_data/gt_maps")
    parser.add_argument(
        "--include_list",
        default="splits/trainval.txt",
        help="Default uses trainval split to avoid test leakage.",
    )
    parser.add_argument(
        "--exclude_list",
        default="splits/test.txt",
        help="Default excludes held-out test IDs.",
    )
    parser.add_argument("--output_root", default="MiaMix/augmented_images")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--k_max", type=int, default=3)
    parser.add_argument("--prob_self", type=float, default=0.5)
    parser.add_argument("--methods", default="mixup,cutmix,agmix")
    parser.add_argument("--method_weights", default="0.33,0.33,0.34")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--warn_only", action="store_true", default=True)
    parser.add_argument("--no_warn_only", action="store_false", dest="warn_only")
    parser.add_argument("--cublas_workspace_config", default=":4096:8")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = _resolve_device(args.device)
    reproducibility_meta = _set_reproducibility(
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        warn_only=bool(args.warn_only),
        cublas_workspace_config=args.cublas_workspace_config,
    )

    output_root = Path(args.output_root)
    output_image_dir = output_root / "images"
    output_label_dir = output_root / "labels"
    output_gt_maps_dir = output_root / "gt_maps"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    output_gt_maps_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(args.heatmap_dir):
        raise FileNotFoundError(f"Heatmap directory not found: {args.heatmap_dir}")

    image_paths = _list_images(args.image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {args.image_dir}")

    include_ids = _load_id_list(args.include_list)
    exclude_ids = _load_id_list(args.exclude_list)
    if include_ids:
        image_paths = [p for p in image_paths if _stem(p) in include_ids]
    if exclude_ids:
        image_paths = [p for p in image_paths if _stem(p) not in exclude_ids]
    if not image_paths:
        raise RuntimeError("No images remain after include/exclude filtering.")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    method_weights = _parse_csv_list(args.method_weights, cast=float)
    if len(methods) != len(method_weights):
        raise ValueError("methods and method_weights must have the same length.")

    print(f"Using device: {device}")
    print(f"Loaded {len(image_paths)} source images after split filtering.")

    image_tf = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor()])
    heatmap_tf = T.Compose(
        [
            T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )

    miamix = MiAMix(
        alpha=args.alpha,
        k_max=args.k_max,
        prob_self=args.prob_self,
        methods=methods,
        method_weights=method_weights,
        device=device,
    ).to(device)

    written_rows = []
    skipped_missing_heatmaps = 0

    for batch_start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[batch_start : batch_start + args.batch_size]
        imgs, hmaps, names, annotator_counts = [], [], [], []

        for path in batch_paths:
            base = _stem(path)
            heatmap_matches = sorted(glob(os.path.join(args.heatmap_dir, f"P*_{base}.png")))
            if not heatmap_matches:
                skipped_missing_heatmaps += 1
                continue

            img = Image.open(path).convert("RGB")
            imgs.append(image_tf(img))

            person_hmaps = []
            for h_match in heatmap_matches:
                h_img = Image.open(h_match).convert("L")
                person_hmaps.append(heatmap_tf(h_img))

            stacked_hmaps = torch.stack(person_hmaps)
            avg_hmap = torch.mean(stacked_hmaps, dim=0)
            hmaps.append(avg_hmap.repeat(args.num_classes, 1, 1))
            names.append(base)
            annotator_counts.append(len(heatmap_matches))

        if not imgs:
            continue

        x = torch.stack(imgs).to(device)
        y = torch.stack(hmaps).to(device)

        with torch.no_grad():
            x_mix, y_mix, self_mix_flags = miamix(x, y)

        for i in range(x_mix.size(0)):
            image_id = f"{names[i]}_AUG"
            save_image(x_mix[i], str(output_image_dir / f"{image_id}.jpg"))
            save_image(y_mix[i][0], str(output_label_dir / f"{image_id}.png"))
            y_np = y_mix[i][0].detach().cpu().numpy()
            y_np = to_prob_map(y_np)
            np.save(str(output_gt_maps_dir / f"{image_id}.npy"), y_np.astype(np.float32))

            mix_type = "SELF-MIX" if self_mix_flags[i] else "CROSS-MIX"
            written_rows.append(
                {
                    "image_id": image_id,
                    "source_image_id": names[i],
                    "mix_type": mix_type,
                    "self_mix": int(bool(self_mix_flags[i])),
                    "annotator_count": int(annotator_counts[i]),
                }
            )

    metadata_csv = output_root / "metadata.csv"
    with open(metadata_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "source_image_id", "mix_type", "self_mix", "annotator_count"],
        )
        writer.writeheader()
        writer.writerows(written_rows)

    run_summary = {
        "image_dir": args.image_dir,
        "heatmap_dir": args.heatmap_dir,
        "include_list": args.include_list,
        "exclude_list": args.exclude_list,
        "output_root": str(output_root),
        "images_after_filtering": len(image_paths),
        "images_written": len(written_rows),
        "skipped_missing_heatmaps": int(skipped_missing_heatmaps),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "num_classes": int(args.num_classes),
        "alpha": float(args.alpha),
        "k_max": int(args.k_max),
        "prob_self": float(args.prob_self),
        "methods": methods,
        "method_weights": [float(x) for x in method_weights],
        "reproducibility": reproducibility_meta,
    }
    with open(output_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(json.dumps(run_summary, indent=2))
    print(f"Wrote metadata: {metadata_csv}")


if __name__ == "__main__":
    main()

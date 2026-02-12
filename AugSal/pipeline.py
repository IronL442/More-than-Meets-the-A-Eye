from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from augsal.backends import create_backend
from augsal.prompting import PromptBuilder
from augsal.pseudo_label import (
    build_pseudo_label,
    compute_change_attention,
    renorm_prob,
    select_saliency_guided_attention_map,
)


def _stem(path: str | Path) -> str:
    return Path(path).stem


def _entropy(prob_map: np.ndarray) -> float:
    p = renorm_prob(prob_map)
    return float(-np.sum(p * np.log(p + 1e-12), dtype=np.float64))


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


def _list_images(image_dir: str) -> List[Path]:
    root = Path(image_dir)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = sorted([p for p in root.iterdir() if p.suffix.lower() in exts and p.is_file()])
    return paths


def _write_image(path: Path, image_rgb: np.ndarray, jpg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if ext in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    elif ext == ".png":
        ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def _resolve_caption(captions: Dict[str, str], file_name: str, stem: str) -> str | None:
    if file_name in captions:
        return captions[file_name]
    if stem in captions:
        return captions[stem]
    file_name_lower = file_name.lower()
    stem_lower = stem.lower()
    for key, value in captions.items():
        if key.lower() == file_name_lower or Path(key).stem.lower() == stem_lower:
            return value
    return None


def _load_gt_map(gt_mean_dir: Path, stem: str) -> np.ndarray:
    npy_path = gt_mean_dir / f"{stem}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing GT mean map: {npy_path}")
    arr = np.load(npy_path).astype(np.float32)
    return renorm_prob(arr)


def _ensure_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Path exists and overwrite=false: {path}")


def _iter_paths(paths: Iterable[Path], progress: bool, desc: str) -> Iterable[Path]:
    if progress and tqdm is not None:
        return tqdm(list(paths), desc=desc)
    return paths


def run(
    cfg_path: str,
    *,
    max_images_override: int | None = None,
    backend_override: str | None = None,
    seed_override: int | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})
    generation_cfg = cfg.get("generation", {})
    prompting_cfg = cfg.get("prompting", {})
    pseudo_cfg = cfg.get("pseudo_label", {})
    cross_cfg = cfg.get("cross_attention", {})
    runtime_cfg = cfg.get("runtime", {})

    image_dir = str(input_cfg.get("image_dir", "data/seminar_data/images"))
    gt_mean_dir = Path(str(input_cfg.get("gt_mean_dir", "data/seminar_data/gt_maps_mean")))
    captions_json = str(input_cfg.get("captions_json", "data/seminar_data/image_captions.json"))
    include_list = input_cfg.get("include_list", "splits/trainval.txt")
    exclude_list = input_cfg.get("exclude_list", None)

    output_root = Path(str(output_cfg.get("root", "AugSal/augmented_data")))
    images_subdir = str(output_cfg.get("images_subdir", "images"))
    gt_subdir = str(output_cfg.get("gt_subdir", "gt_maps"))
    metadata_csv_name = str(output_cfg.get("metadata_csv", "metadata.csv"))
    metadata_jsonl_name = str(output_cfg.get("metadata_jsonl", "metadata.jsonl"))
    copy_originals = bool(output_cfg.get("copy_originals", True))
    save_change_maps = bool(output_cfg.get("save_change_maps", True))
    change_maps_subdir = str(output_cfg.get("change_maps_subdir", "change_maps"))
    overwrite = bool(output_cfg.get("overwrite", True))

    cross_enabled = bool(cross_cfg.get("enabled", False))
    cross_use_for_pseudo = bool(cross_cfg.get("use_for_pseudo_label", True))
    cross_blend_weight = float(cross_cfg.get("blend_weight", 0.5))
    cross_blend_weight = float(np.clip(cross_blend_weight, 0.0, 1.0))
    cross_min_token_chars = int(cross_cfg.get("min_token_chars", 2))
    cross_save_selected_maps = bool(cross_cfg.get("save_selected_maps", False))
    cross_selected_maps_subdir = str(cross_cfg.get("selected_maps_subdir", "selected_attention_maps"))

    seed = int(seed_override if seed_override is not None else generation_cfg.get("seed", 1337))
    rng = np.random.default_rng(seed)

    if backend_override:
        generation_cfg = dict(generation_cfg)
        generation_cfg["backend"] = backend_override

    backend = create_backend(generation_cfg)
    prompt_builder = PromptBuilder(
        negative_prompt=str(prompting_cfg.get("negative_prompt", "")),
    )

    num_augs_per_image = int(generation_cfg.get("num_augs_per_image", 1))
    if num_augs_per_image < 1:
        raise ValueError("generation.num_augs_per_image must be >= 1")

    image_ext = str(generation_cfg.get("image_ext", ".jpg"))
    if not image_ext.startswith("."):
        image_ext = f".{image_ext}"
    jpg_quality = int(generation_cfg.get("jpg_quality", 95))

    max_images = max_images_override if max_images_override is not None else runtime_cfg.get("max_images", None)
    max_images = None if max_images is None else int(max_images)
    progress = bool(runtime_cfg.get("progress", True))

    with open(captions_json, "r", encoding="utf-8") as f:
        captions_raw = json.load(f)
    captions = {str(k): str(v) for k, v in captions_raw.items()}

    include_ids = _load_id_list(str(include_list) if include_list else None)
    exclude_ids = _load_id_list(str(exclude_list) if exclude_list else None)

    all_images = _list_images(image_dir)
    if not all_images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    filtered_images: List[Path] = []
    for path in all_images:
        sid = path.stem
        if include_ids and sid not in include_ids:
            continue
        if exclude_ids and sid in exclude_ids:
            continue
        filtered_images.append(path)

    if max_images is not None:
        filtered_images = filtered_images[:max_images]

    if not filtered_images:
        raise ValueError("No images remain after include/exclude/max_images filtering.")

    images_out_dir = output_root / images_subdir
    gt_out_dir = output_root / gt_subdir
    change_out_dir = output_root / change_maps_subdir
    cross_selected_out_dir = output_root / cross_selected_maps_subdir
    metadata_csv_path = output_root / metadata_csv_name
    metadata_jsonl_path = output_root / metadata_jsonl_name

    if not dry_run:
        images_out_dir.mkdir(parents=True, exist_ok=True)
        gt_out_dir.mkdir(parents=True, exist_ok=True)
        if save_change_maps:
            change_out_dir.mkdir(parents=True, exist_ok=True)
        if cross_enabled and cross_save_selected_maps:
            cross_selected_out_dir.mkdir(parents=True, exist_ok=True)
        _ensure_empty_or_overwrite(metadata_csv_path, overwrite)
        _ensure_empty_or_overwrite(metadata_jsonl_path, overwrite)

    rows: List[Dict[str, Any]] = []
    missing_caption_count = 0

    iterator = _iter_paths(filtered_images, progress=progress, desc="AugSal")
    for img_path in iterator:
        stem = img_path.stem
        file_name = img_path.name

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gt_map = _load_gt_map(gt_mean_dir, stem)
        gt_entropy = _entropy(gt_map)

        caption = _resolve_caption(captions, file_name=file_name, stem=stem)
        if caption is None:
            caption = stem.replace("_", " ")
            missing_caption_count += 1

        if copy_originals:
            out_name = f"{stem}{image_ext}"
            out_image_path = images_out_dir / out_name
            out_gt_path = gt_out_dir / f"{stem}.npy"
            if not dry_run:
                _write_image(out_image_path, img_rgb, jpg_quality=jpg_quality)
                np.save(out_gt_path, gt_map.astype(np.float32))
            rows.append(
                {
                    "image_id": stem,
                    "source_image_id": stem,
                    "image_file": out_name,
                    "gt_file": f"{stem}.npy",
                    "is_augmented": 0,
                    "augmentation_index": -1,
                    "backend": "original_copy",
                    "seed": "",
                    "caption": caption,
                    "prompt": "",
                    "negative_prompt": "",
                    "style_tag": "",
                    "mean_abs_pixel_change": 0.0,
                    "max_abs_pixel_change": 0.0,
                    "change_entropy": 0.0,
                    "cross_attention_used": 0,
                    "cross_attention_num_maps": 0,
                    "cross_attention_map_entropy": 0.0,
                    "selected_token_index": -1,
                    "selected_token": "",
                    "selected_token_score": 0.0,
                    "gt_entropy": gt_entropy,
                    "pseudo_gt_entropy": gt_entropy,
                }
            )

        for aug_idx in range(num_augs_per_image):
            aug_seed = int(rng.integers(0, 2**31 - 1))
            local_rng = np.random.default_rng(aug_seed)
            prompt = prompt_builder.build(caption, local_rng)

            aug_id = f"{stem}_AUG{aug_idx:03d}"
            aug_image_name = f"{aug_id}{image_ext}"
            aug_gt_name = f"{aug_id}.npy"

            if hasattr(backend, "generate_with_aux"):
                aug_rgb, aug_aux = backend.generate_with_aux(
                    img_rgb,
                    prompt=prompt.positive,
                    negative_prompt=prompt.negative,
                    caption=caption,
                    seed=aug_seed,
                )
            else:
                aug_rgb = backend.generate(
                    img_rgb,
                    prompt=prompt.positive,
                    negative_prompt=prompt.negative,
                    caption=caption,
                    seed=aug_seed,
                )
                aug_aux = {}
            if aug_rgb.shape != img_rgb.shape:
                aug_rgb = cv2.resize(
                    aug_rgb,
                    (img_rgb.shape[1], img_rgb.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            change = compute_change_attention(
                img_rgb,
                aug_rgb,
                blur_ksize=int(pseudo_cfg.get("change_blur_ksize", 17)),
                blur_sigma=float(pseudo_cfg.get("change_blur_sigma", 3.0)),
                min_change_threshold=float(pseudo_cfg.get("min_change_threshold", 2.0)),
            )
            change_for_pseudo = change
            cross_attention_used = 0
            cross_attention_num_maps = 0
            cross_attention_map_entropy = 0.0
            selected_token = ""
            selected_token_index = -1
            selected_token_score = 0.0

            if cross_enabled and cross_use_for_pseudo and isinstance(aug_aux, dict):
                cross_data = aug_aux.get("cross_attention", {})
                if isinstance(cross_data, dict):
                    token_maps = cross_data.get("token_attention_maps", None)
                    token_texts = cross_data.get("tokens", [])
                    if isinstance(token_maps, np.ndarray) and token_maps.ndim == 3 and token_maps.shape[0] > 0:
                        sel = select_saliency_guided_attention_map(
                            token_maps,
                            token_texts,
                            gt_map,
                            min_token_chars=cross_min_token_chars,
                        )
                        selected_map = sel["selected_map"]
                        selected_token = str(sel["selected_token"])
                        selected_token_index = int(sel["selected_index"])
                        selected_token_score = float(sel["selected_score"])
                        cross_attention_num_maps = int(sel["num_maps"])
                        if selected_token_index >= 0:
                            cross_attention_used = 1
                            cross_attention_map_entropy = _entropy(selected_map)
                        change_for_pseudo = renorm_prob(
                            (1.0 - cross_blend_weight) * change + cross_blend_weight * selected_map
                        )
                        if (
                            cross_attention_used
                            and cross_save_selected_maps
                            and not dry_run
                        ):
                            np.save(
                                cross_selected_out_dir / f"{aug_id}.npy",
                                selected_map.astype(np.float32),
                            )

            pseudo = build_pseudo_label(
                gt_map,
                change_for_pseudo,
                diff_weight=float(pseudo_cfg.get("diff_weight", 0.35)),
                change_floor=float(pseudo_cfg.get("change_floor", 1e-6)),
                smooth_ksize=int(pseudo_cfg.get("smooth_ksize", 9)),
                smooth_sigma=float(pseudo_cfg.get("smooth_sigma", 2.0)),
            )

            delta = np.abs(img_rgb.astype(np.float32) - aug_rgb.astype(np.float32))
            mean_abs_change = float(delta.mean())
            max_abs_change = float(delta.max())
            change_entropy = _entropy(change)
            pseudo_entropy = _entropy(pseudo)

            if not dry_run:
                _write_image(images_out_dir / aug_image_name, aug_rgb, jpg_quality=jpg_quality)
                np.save(gt_out_dir / aug_gt_name, pseudo.astype(np.float32))
                if save_change_maps:
                    np.save(change_out_dir / f"{aug_id}.npy", change.astype(np.float32))

            rows.append(
                {
                    "image_id": aug_id,
                    "source_image_id": stem,
                    "image_file": aug_image_name,
                    "gt_file": aug_gt_name,
                    "is_augmented": 1,
                    "augmentation_index": aug_idx,
                    "backend": backend.name,
                    "seed": aug_seed,
                    "caption": caption,
                    "prompt": prompt.positive,
                    "negative_prompt": prompt.negative,
                    "style_tag": prompt.style_tag,
                    "mean_abs_pixel_change": mean_abs_change,
                    "max_abs_pixel_change": max_abs_change,
                    "change_entropy": change_entropy,
                    "cross_attention_used": cross_attention_used,
                    "cross_attention_num_maps": cross_attention_num_maps,
                    "cross_attention_map_entropy": cross_attention_map_entropy,
                    "selected_token_index": selected_token_index,
                    "selected_token": selected_token,
                    "selected_token_score": selected_token_score,
                    "gt_entropy": gt_entropy,
                    "pseudo_gt_entropy": pseudo_entropy,
                }
            )

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with open(metadata_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with open(metadata_jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        run_summary = {
            "config": cfg_path,
            "seed": seed,
            "backend": backend.name,
            "source_images": len(filtered_images),
            "num_augs_per_image": num_augs_per_image,
            "output_rows": len(rows),
            "copy_originals": copy_originals,
            "missing_caption_count": missing_caption_count,
            "cross_attention_enabled": cross_enabled,
            "cross_attention_use_for_pseudo_label": cross_use_for_pseudo,
            "cross_attention_blend_weight": cross_blend_weight,
        }
        with open(output_root / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)
    else:
        run_summary = {
            "config": cfg_path,
            "seed": seed,
            "backend": backend.name,
            "source_images": len(filtered_images),
            "num_augs_per_image": num_augs_per_image,
            "output_rows": len(rows),
            "copy_originals": copy_originals,
            "missing_caption_count": missing_caption_count,
            "cross_attention_enabled": cross_enabled,
            "cross_attention_use_for_pseudo_label": cross_use_for_pseudo,
            "cross_attention_blend_weight": cross_blend_weight,
            "dry_run": True,
        }

    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AugSal-like synthetic training data.")
    parser.add_argument("--config", type=str, default="AugSal/configs/default.yaml")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    summary = run(
        args.config,
        max_images_override=args.max_images,
        backend_override=args.backend,
        seed_override=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

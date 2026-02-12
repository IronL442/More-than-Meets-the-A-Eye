from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np


def renorm_prob(arr: np.ndarray, *, eps: float = 0.0) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    out = np.clip(out, 0.0, None)
    if eps > 0.0:
        out = out + np.float32(eps)
    s = float(out.sum())
    if s <= 0.0:
        h, w = out.shape
        return np.full((h, w), 1.0 / float(h * w), dtype=np.float32)
    return (out / s).astype(np.float32, copy=False)


def _odd_ksize(ksize: int) -> int:
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return k


def compute_change_attention(
    image_rgb: np.ndarray,
    aug_rgb: np.ndarray,
    *,
    blur_ksize: int = 17,
    blur_sigma: float = 3.0,
    min_change_threshold: float = 2.0,
) -> np.ndarray:
    """Returns a probability map highlighting where augmentation changed the image."""
    if image_rgb.shape != aug_rgb.shape:
        raise ValueError("compute_change_attention expects equal image shapes.")

    orig = image_rgb.astype(np.float32)
    aug = aug_rgb.astype(np.float32)
    diff = np.mean(np.abs(orig - aug), axis=2)

    if min_change_threshold > 0.0:
        diff = np.clip(diff - float(min_change_threshold), 0.0, None)

    k = _odd_ksize(blur_ksize)
    diff = cv2.GaussianBlur(diff, (k, k), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))
    return renorm_prob(diff)


def build_pseudo_label(
    gt_map: np.ndarray,
    change_attention: np.ndarray,
    *,
    diff_weight: float = 0.35,
    change_floor: float = 1e-6,
    smooth_ksize: int = 9,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """Blend GT map with augmentation-change attention to form pseudo saliency."""
    gt = renorm_prob(gt_map)
    change = renorm_prob(change_attention, eps=float(change_floor))

    guided = renorm_prob(gt * change)

    w = float(np.clip(diff_weight, 0.0, 1.0))
    pseudo = renorm_prob((1.0 - w) * gt + w * guided)

    k = _odd_ksize(smooth_ksize)
    if k > 1:
        pseudo = cv2.GaussianBlur(pseudo, (k, k), sigmaX=float(smooth_sigma), sigmaY=float(smooth_sigma))
        pseudo = renorm_prob(pseudo)

    return pseudo


def _resize_prob_map(m: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    arr = np.asarray(m, dtype=np.float32)
    if arr.shape == (h, w):
        return renorm_prob(arr)
    resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return renorm_prob(resized)


def _clean_token_text(token: str) -> str:
    text = str(token)
    text = text.replace("Ġ", "").replace("▁", "").replace("</w>", "")
    text = text.strip()
    return text


def _is_valid_token(
    token: str,
    *,
    min_token_chars: int,
    ignore_tokens: set[str],
) -> bool:
    t = _clean_token_text(token)
    if not t:
        return False
    if t in ignore_tokens:
        return False
    if len(t) < int(min_token_chars):
        return False
    return any(ch.isalnum() for ch in t)


def select_saliency_guided_attention_map(
    token_attention_maps: np.ndarray,
    token_texts: Sequence[str],
    gt_map: np.ndarray,
    *,
    min_token_chars: int = 2,
    ignore_tokens: Sequence[str] = ("<s>", "</s>", "<pad>", "<|endoftext|>"),
) -> dict[str, Any]:
    """Select the token attention map with maximal overlap with GT saliency."""
    maps = np.asarray(token_attention_maps, dtype=np.float32)
    gt = renorm_prob(gt_map)

    default_out = {
        "selected_map": gt,
        "selected_index": -1,
        "selected_token": "",
        "selected_score": 0.0,
        "num_candidates": 0,
        "num_maps": int(maps.shape[0]) if maps.ndim == 3 else 0,
    }

    if maps.ndim != 3 or maps.shape[0] == 0:
        return default_out

    h_attn, w_attn = maps.shape[1], maps.shape[2]
    gt_small = _resize_prob_map(gt, (h_attn, w_attn))
    ignore_set = set(str(t) for t in ignore_tokens)

    best_idx = -1
    best_score = -1.0
    best_map_small: np.ndarray | None = None
    candidate_count = 0

    n_maps = int(maps.shape[0])
    n_tokens = len(token_texts)
    n = min(n_maps, n_tokens) if n_tokens > 0 else n_maps
    for idx in range(n):
        token = token_texts[idx] if idx < n_tokens else ""
        if n_tokens > 0 and not _is_valid_token(
            token,
            min_token_chars=min_token_chars,
            ignore_tokens=ignore_set,
        ):
            continue
        m_small = renorm_prob(maps[idx])
        score = float(np.sum(m_small * gt_small))
        candidate_count += 1
        if score > best_score:
            best_score = score
            best_idx = idx
            best_map_small = m_small

    if best_map_small is None:
        avg_small = renorm_prob(np.mean(maps[:n], axis=0))
        selected_map = _resize_prob_map(avg_small, gt.shape)
        default_out["selected_map"] = selected_map
        return default_out

    selected_map = _resize_prob_map(best_map_small, gt.shape)
    selected_token = token_texts[best_idx] if best_idx < n_tokens else ""
    return {
        "selected_map": selected_map,
        "selected_index": int(best_idx),
        "selected_token": _clean_token_text(selected_token),
        "selected_score": float(best_score),
        "num_candidates": int(candidate_count),
        "num_maps": int(n_maps),
    }

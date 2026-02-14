from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict

import cv2
import numpy as np


class AugmentationBackend:
    name = "base"

    def generate(
        self,
        image_rgb: np.ndarray,
        *,
        prompt: str,
        negative_prompt: str,
        caption: str,
        seed: int,
        saliency_map: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def generate_with_aux(
        self,
        image_rgb: np.ndarray,
        *,
        prompt: str,
        negative_prompt: str,
        caption: str,
        seed: int,
        saliency_map: np.ndarray | None = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        return self.generate(
            image_rgb,
            prompt=prompt,
            negative_prompt=negative_prompt,
            caption=caption,
            seed=seed,
            saliency_map=saliency_map,
        ), {}


@dataclass
class OpenCVCaptionStyleConfig:
    max_rotation_deg: float = 7.0
    max_shift_ratio: float = 0.04
    min_scale: float = 0.96
    max_scale: float = 1.04
    perspective_prob: float = 0.35
    perspective_jitter_ratio: float = 0.03
    brightness_delta: float = 20.0
    contrast_delta: float = 0.18
    saturation_delta: float = 0.15
    noise_std: float = 3.0
    saliency_only: bool = True
    saliency_min: float = 0.15
    saliency_gamma: float = 1.25
    saliency_blur_ksize: int = 31
    saliency_blend: float = 1.0


class OpenCVCaptionStyleBackend(AugmentationBackend):
    """Lightweight, dependency-free image editing backend.

    It approximates "text-guided" variation by using caption keywords to bias
    photometric transforms while keeping scene geometry mostly intact.
    """

    name = "opencv_caption_style"

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        raw = cfg or {}
        self.cfg = OpenCVCaptionStyleConfig(
            max_rotation_deg=float(raw.get("max_rotation_deg", 7.0)),
            max_shift_ratio=float(raw.get("max_shift_ratio", 0.04)),
            min_scale=float(raw.get("min_scale", 0.96)),
            max_scale=float(raw.get("max_scale", 1.04)),
            perspective_prob=float(raw.get("perspective_prob", 0.35)),
            perspective_jitter_ratio=float(raw.get("perspective_jitter_ratio", 0.03)),
            brightness_delta=float(raw.get("brightness_delta", 20.0)),
            contrast_delta=float(raw.get("contrast_delta", 0.18)),
            saturation_delta=float(raw.get("saturation_delta", 0.15)),
            noise_std=float(raw.get("noise_std", 3.0)),
            saliency_only=bool(raw.get("saliency_only", True)),
            saliency_min=float(raw.get("saliency_min", 0.15)),
            saliency_gamma=float(raw.get("saliency_gamma", 1.25)),
            saliency_blur_ksize=int(raw.get("saliency_blur_ksize", 31)),
            saliency_blend=float(raw.get("saliency_blend", 1.0)),
        )

    @staticmethod
    def _keyword_flags(caption: str) -> Dict[str, bool]:
        text = caption.lower()
        return {
            "dark": any(k in text for k in ("night", "dark", "dim", "evening")),
            "bright": any(k in text for k in ("bright", "sunny", "daylight", "light")),
            "indoor": any(k in text for k in ("indoor", "office", "room", "classroom")),
            "outdoor": any(k in text for k in ("outdoor", "street", "park", "building", "sky")),
            "person": any(k in text for k in ("person", "man", "woman", "people", "student")),
        }

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(k in text for k in keywords)

    def _prompt_controls(self, prompt: str, negative_prompt: str) -> Dict[str, float]:
        text = str(prompt).lower()
        neg = str(negative_prompt).lower()

        controls: Dict[str, float] = {
            "extra_shift_ratio": 0.0,
            "extra_zoom": 0.0,
            "extra_perspective_prob": 0.0,
            "brightness_bias": 0.0,
            "contrast_bias": 0.0,
            "saturation_bias": 0.0,
            "warmth_bias": 0.0,
            "noise_multiplier": 1.0,
            "person_blur_prob_bias": 0.0,
        }

        if self._contains_any(text, ("shifted viewpoint", "framing change", "viewpoint")):
            controls["extra_shift_ratio"] += 0.02
        if self._contains_any(text, ("zoom variation", "zoom")):
            controls["extra_zoom"] += 0.02
        if self._contains_any(text, ("focal-length variation", "focal length")):
            controls["extra_perspective_prob"] += 0.15

        if "high dynamic range" in text:
            controls["contrast_bias"] += 0.08
        if "soft diffused lighting" in text:
            controls["contrast_bias"] -= 0.05
            controls["brightness_bias"] += 5.0
        if "studio-like lighting" in text:
            controls["contrast_bias"] += 0.03
            controls["saturation_bias"] -= 0.03
        if "cinematic natural lighting" in text:
            controls["contrast_bias"] += 0.04
            controls["saturation_bias"] += 0.02
        if "warm color grading" in text:
            controls["warmth_bias"] += 1.0
        if "cool color grading" in text:
            controls["warmth_bias"] -= 1.0

        if self._contains_any(neg, ("surreal", "distorted", "artifact", "cartoon")):
            controls["noise_multiplier"] *= 0.6
            controls["person_blur_prob_bias"] -= 0.15

        return controls

    def _affine_warp(
        self,
        image_rgb: np.ndarray,
        rng: np.random.Generator,
        controls: Dict[str, float],
    ) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        shift_ratio = float(np.clip(
            self.cfg.max_shift_ratio + controls["extra_shift_ratio"],
            0.0,
            0.25,
        ))
        min_scale = max(0.5, self.cfg.min_scale - controls["extra_zoom"])
        max_scale = min(1.5, self.cfg.max_scale + controls["extra_zoom"])
        if max_scale < min_scale:
            max_scale = min_scale

        angle = float(rng.uniform(-self.cfg.max_rotation_deg, self.cfg.max_rotation_deg))
        scale = float(rng.uniform(min_scale, max_scale))
        tx = float(rng.uniform(-shift_ratio, shift_ratio) * w)
        ty = float(rng.uniform(-shift_ratio, shift_ratio) * h)

        m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
        m[:, 2] += [tx, ty]
        warped = cv2.warpAffine(
            image_rgb,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        perspective_prob = float(np.clip(
            self.cfg.perspective_prob + controls["extra_perspective_prob"],
            0.0,
            1.0,
        ))
        if float(rng.random()) < perspective_prob:
            jitter = self.cfg.perspective_jitter_ratio
            src = np.array(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                dtype=np.float32,
            )
            dx = jitter * w
            dy = jitter * h
            dst = src + np.array(
                [
                    [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
                    [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
                    [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
                    [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
                ],
                dtype=np.float32,
            )
            p = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(
                warped,
                p,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )

        return warped

    def _photometric_edit(
        self,
        image_rgb: np.ndarray,
        rng: np.random.Generator,
        flags: Dict[str, bool],
        controls: Dict[str, float],
    ) -> np.ndarray:
        out = image_rgb.astype(np.float32)

        contrast = 1.0 + float(rng.uniform(-self.cfg.contrast_delta, self.cfg.contrast_delta))
        contrast += controls["contrast_bias"]
        contrast = float(np.clip(contrast, 0.6, 1.6))
        brightness = float(rng.uniform(-self.cfg.brightness_delta, self.cfg.brightness_delta))
        brightness += controls["brightness_bias"]

        if flags["dark"]:
            brightness -= 10.0
            contrast += 0.04
        elif flags["bright"]:
            brightness += 10.0

        out = out * contrast + brightness

        warmth = controls["warmth_bias"]
        if warmth != 0.0:
            temp_shift = 10.0 * float(warmth)
            out[:, :, 0] += temp_shift
            out[:, :, 2] -= temp_shift

        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_scale = 1.0 + float(rng.uniform(-self.cfg.saturation_delta, self.cfg.saturation_delta))
        sat_scale += controls["saturation_bias"]
        if flags["outdoor"]:
            sat_scale += 0.05
        if flags["indoor"]:
            sat_scale -= 0.03
        sat_scale = float(np.clip(sat_scale, 0.6, 1.6))
        hsv[:, :, 1] *= sat_scale
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        person_blur_prob = float(np.clip(0.5 + controls["person_blur_prob_bias"], 0.0, 1.0))
        if flags["person"] and float(rng.random()) < person_blur_prob:
            out = cv2.GaussianBlur(out, (3, 3), sigmaX=0.6)

        noise_std = self.cfg.noise_std * max(0.0, controls["noise_multiplier"])
        noise = rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 0, 255)
        return out.astype(np.uint8)

    def _prepare_saliency_mask(
        self,
        saliency_map: np.ndarray | None,
        shape_hw: tuple[int, int],
    ) -> np.ndarray | None:
        if saliency_map is None:
            return None

        h, w = shape_hw
        arr = np.asarray(saliency_map, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.ndim != 2:
            return None
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

        arr = arr - float(arr.min())
        maxv = float(arr.max())
        if maxv <= 1e-8:
            return None
        arr = arr / maxv

        smin = float(np.clip(self.cfg.saliency_min, 0.0, 0.95))
        arr = np.clip((arr - smin) / max(1e-6, 1.0 - smin), 0.0, 1.0)

        gamma = max(0.1, float(self.cfg.saliency_gamma))
        arr = np.power(arr, gamma).astype(np.float32)

        k = int(self.cfg.saliency_blur_ksize)
        if k % 2 == 0:
            k += 1
        if k >= 3:
            arr = cv2.GaussianBlur(arr, (k, k), sigmaX=0.0)
        arr = np.clip(arr, 0.0, 1.0)

        blend = float(np.clip(self.cfg.saliency_blend, 0.0, 1.0))
        arr = np.clip(arr * blend, 0.0, 1.0)
        if float(arr.max()) <= 1e-6:
            return None
        return arr[:, :, None]

    def generate(
        self,
        image_rgb: np.ndarray,
        *,
        prompt: str,
        negative_prompt: str,
        caption: str,
        seed: int,
        saliency_map: np.ndarray | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        flags = self._keyword_flags(caption)
        controls = self._prompt_controls(prompt, negative_prompt)

        if self.cfg.saliency_only:
            mask = self._prepare_saliency_mask(saliency_map, image_rgb.shape[:2])
            if mask is not None:
                edited = self._photometric_edit(image_rgb, rng, flags, controls).astype(np.float32)
                base = image_rgb.astype(np.float32)
                out = np.clip(base * (1.0 - mask) + edited * mask, 0.0, 255.0)
                return out.astype(np.uint8)

        out = self._affine_warp(image_rgb, rng, controls)
        out = self._photometric_edit(out, rng, flags, controls)
        return out


class _CrossAttentionRecorder:
    def __init__(
        self,
        torch_module: Any,
        *,
        map_hw: tuple[int, int] = (64, 64),
        layer_name_contains: str = "attn2",
    ) -> None:
        self._torch = torch_module
        self.map_hw = (int(map_hw[0]), int(map_hw[1]))
        self.layer_name_contains = str(layer_name_contains)
        self._sum_maps = None
        self._num_records = 0

    @staticmethod
    def _infer_hw(q_len: int) -> tuple[int, int] | None:
        side = int(round(math.sqrt(float(q_len))))
        if side * side == int(q_len):
            return side, side
        return None

    def reset(self) -> None:
        self._sum_maps = None
        self._num_records = 0

    def should_record(self, module_name: str, is_cross: bool) -> bool:
        if not is_cross:
            return False
        if self.layer_name_contains and self.layer_name_contains not in module_name:
            return False
        return True

    def record(
        self,
        attention_probs: Any,
        *,
        batch_size: int,
        spatial_hw: tuple[int, int] | None,
        module_name: str,
    ) -> None:
        if batch_size <= 0:
            return
        if not self.should_record(module_name, is_cross=True):
            return

        q_len = int(attention_probs.shape[1])
        k_len = int(attention_probs.shape[2])
        if spatial_hw is None or spatial_hw[0] * spatial_hw[1] != q_len:
            inferred = self._infer_hw(q_len)
            if inferred is None:
                return
            h, w = inferred
        else:
            h, w = spatial_hw

        heads = int(attention_probs.shape[0]) // int(batch_size)
        if heads <= 0:
            return

        probs = attention_probs.reshape(batch_size, heads, q_len, k_len)
        probs = probs.mean(dim=(0, 1))  # [q_len, k_len]
        maps = probs.reshape(h, w, k_len).permute(2, 0, 1).contiguous()  # [k_len, h, w]
        maps = maps.to(dtype=self._torch.float32)
        maps = self._torch.nn.functional.interpolate(
            maps.unsqueeze(0),
            size=self.map_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        maps = maps.detach().cpu()

        if self._sum_maps is None:
            self._sum_maps = maps
        else:
            common_k = min(int(self._sum_maps.shape[0]), int(maps.shape[0]))
            self._sum_maps = self._sum_maps[:common_k] + maps[:common_k]
        self._num_records += 1

    def export(
        self,
        *,
        token_texts: list[str],
        token_ids: list[int],
        include_special_tokens: bool,
        max_token_maps: int,
    ) -> Dict[str, Any]:
        if self._sum_maps is None or self._num_records <= 0:
            return {}

        mean_maps = (self._sum_maps / float(self._num_records)).numpy().astype(np.float32)
        num_maps = int(mean_maps.shape[0])
        if max_token_maps > 0:
            num_maps = min(num_maps, int(max_token_maps))
        mean_maps = mean_maps[:num_maps]

        n_tokens = len(token_texts)
        n_ids = len(token_ids)
        n = min(num_maps, n_tokens if n_tokens > 0 else num_maps, n_ids if n_ids > 0 else num_maps)
        mean_maps = mean_maps[:n]
        tok = token_texts[:n] if n_tokens > 0 else ["" for _ in range(n)]
        ids = token_ids[:n] if n_ids > 0 else [-1 for _ in range(n)]

        if not include_special_tokens:
            keep_idx: list[int] = []
            special = {"<s>", "</s>", "<pad>", "<|endoftext|>"}
            for i, t in enumerate(tok):
                raw = str(t).strip()
                if raw in special or not raw:
                    continue
                keep_idx.append(i)
            if keep_idx:
                mean_maps = mean_maps[keep_idx]
                tok = [tok[i] for i in keep_idx]
                ids = [ids[i] for i in keep_idx]

        return {
            "token_attention_maps": mean_maps,
            "tokens": [str(t) for t in tok],
            "token_ids": [int(v) for v in ids],
            "records": int(self._num_records),
            "map_hw": [int(self.map_hw[0]), int(self.map_hw[1])],
        }


class _RecordingCrossAttnProcessor:
    def __init__(
        self,
        *,
        recorder: _CrossAttentionRecorder,
        module_name: str,
    ) -> None:
        self.recorder = recorder
        self.module_name = module_name

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        temb: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del temb, args, kwargs
        residual = hidden_states
        input_ndim = hidden_states.ndim
        spatial_hw: tuple[int, int] | None = None

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            spatial_hw = (int(height), int(width))
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = int(hidden_states.shape[0])

        key_states = encoder_hidden_states
        is_cross = key_states is not None
        if key_states is None:
            key_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            key_states = attn.norm_encoder_hidden_states(key_states)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(key_states)
        value = attn.to_v(key_states)

        batch_size = int(hidden_states.shape[0])
        key_tokens = int(key.shape[1])
        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if self.recorder.should_record(self.module_name, is_cross=is_cross):
            self.recorder.record(
                attention_probs,
                batch_size=batch_size,
                spatial_hw=spatial_hw,
                module_name=self.module_name,
            )

        hidden_states = self.recorder._torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            batch_size, channel, height, width = residual.shape
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


class DiffusersImg2ImgBackend(AugmentationBackend):
    """Optional backend using diffusers image-to-image generation."""

    name = "diffusers_img2img"

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        raw = cfg or {}
        self.model_id = str(raw.get("model_id", "stabilityai/sdxl-turbo"))
        self.strength = float(raw.get("strength", 0.35))
        self.guidance_scale = float(raw.get("guidance_scale", 5.0))
        self.num_inference_steps = int(raw.get("num_inference_steps", 20))
        self.device = str(raw.get("device", "auto"))
        self.max_side = int(raw.get("max_side", 0) or 0)
        self.enable_attention_slicing = bool(raw.get("enable_attention_slicing", False))
        self.enable_vae_slicing = bool(raw.get("enable_vae_slicing", False))
        self.enable_vae_tiling = bool(raw.get("enable_vae_tiling", False))
        self.enable_model_cpu_offload = bool(raw.get("enable_model_cpu_offload", False))
        self.enable_sequential_cpu_offload = bool(raw.get("enable_sequential_cpu_offload", False))
        self.enable_xformers_memory_efficient_attention = bool(
            raw.get("enable_xformers_memory_efficient_attention", False)
        )
        cross_cfg = raw.get("cross_attention", {}) or {}
        self.capture_cross_attention = bool(cross_cfg.get("enabled", False))
        map_hw_raw = cross_cfg.get("map_hw", [64, 64])
        if not isinstance(map_hw_raw, (list, tuple)) or len(map_hw_raw) != 2:
            raise ValueError("diffusers.cross_attention.map_hw must be a 2-item list [H, W].")
        self.capture_map_hw = (int(map_hw_raw[0]), int(map_hw_raw[1]))
        if self.capture_map_hw[0] <= 0 or self.capture_map_hw[1] <= 0:
            raise ValueError("diffusers.cross_attention.map_hw values must be positive.")
        self.capture_layer_name_contains = str(cross_cfg.get("layer_name_contains", "attn2"))
        self.capture_include_special_tokens = bool(cross_cfg.get("include_special_tokens", False))
        self.capture_max_token_maps = int(cross_cfg.get("max_token_maps", 77))

        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Diffusers backend requires `torch` and `diffusers`. "
                "Install them or switch generation.backend to opencv_caption_style."
            ) from exc

        self._torch = torch
        dtype_name = str(raw.get("torch_dtype", "float16")).lower()
        if dtype_name == "float16":
            dtype = torch.float16
        elif dtype_name == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Newer diffusers versions prefer `dtype`; keep fallback for older versions.
        try:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_id,
                dtype=dtype,
                use_safetensors=bool(raw.get("use_safetensors", True)),
            )
        except TypeError:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                use_safetensors=bool(raw.get("use_safetensors", True)),
            )

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        self.device = device
        if str(device).startswith("cuda"):
            if self.enable_sequential_cpu_offload:
                self.pipe.enable_sequential_cpu_offload()
            elif self.enable_model_cpu_offload:
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(device)

            if self.enable_attention_slicing and hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            if self.enable_vae_slicing and hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
            if self.enable_vae_tiling and hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()
            if self.enable_xformers_memory_efficient_attention and hasattr(
                self.pipe, "enable_xformers_memory_efficient_attention"
            ):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
        else:
            self.pipe.to(device)

        self._recorder: _CrossAttentionRecorder | None = None
        if self.capture_cross_attention:
            self._recorder = _CrossAttentionRecorder(
                torch,
                map_hw=self.capture_map_hw,
                layer_name_contains=self.capture_layer_name_contains,
            )
            self._install_cross_attention_processors()

    def _install_cross_attention_processors(self) -> None:
        if self._recorder is None:
            return
        processors: Dict[str, Any] = {}
        for name in self.pipe.unet.attn_processors.keys():
            processors[name] = _RecordingCrossAttnProcessor(
                recorder=self._recorder,
                module_name=name,
            )
        self.pipe.unet.set_attn_processor(processors)

    def _tokenize_prompt(self, prompt: str) -> tuple[list[int], list[str]]:
        tokenizer = getattr(self.pipe, "tokenizer", None)
        if tokenizer is None:
            return [], []

        max_length = int(getattr(tokenizer, "model_max_length", 77))
        tok = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = tok.input_ids[0].tolist()

        tokens: list[str] = []
        convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
        if callable(convert_ids_to_tokens):
            try:
                tokens = [str(t) for t in convert_ids_to_tokens(ids)]
            except Exception:
                tokens = []
        if not tokens:
            tokens = [str(tokenizer.decode([int(i)])) for i in ids]
        return [int(i) for i in ids], tokens

    def _prepare_img2img_input(self, image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = image_rgb.shape[:2]
        if self.max_side <= 0 or max(h, w) <= self.max_side:
            return image_rgb, (h, w)

        scale = float(self.max_side) / float(max(h, w))
        new_h = max(64, int(round((h * scale) / 8.0) * 8))
        new_w = max(64, int(round((w * scale) / 8.0) * 8))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, (h, w)

    def generate_with_aux(
        self,
        image_rgb: np.ndarray,
        *,
        prompt: str,
        negative_prompt: str,
        caption: str,
        seed: int,
        saliency_map: np.ndarray | None = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        del caption, saliency_map
        from PIL import Image

        if self.capture_cross_attention and self._recorder is not None:
            self._recorder.reset()

        proc_rgb, (h, w) = self._prepare_img2img_input(image_rgb)
        pil_image = Image.fromarray(proc_rgb)

        generator = self._torch.Generator(device=self.device)
        generator.manual_seed(int(seed))

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            strength=self.strength,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        ).images[0]

        out = np.asarray(out.resize((w, h), Image.BICUBIC).convert("RGB"), dtype=np.uint8)
        aux: Dict[str, Any] = {}

        if self.capture_cross_attention and self._recorder is not None:
            token_ids, token_texts = self._tokenize_prompt(prompt)
            cross_payload = self._recorder.export(
                token_texts=token_texts,
                token_ids=token_ids,
                include_special_tokens=self.capture_include_special_tokens,
                max_token_maps=self.capture_max_token_maps,
            )
            if cross_payload:
                aux["cross_attention"] = cross_payload

        return out, aux

    def generate(
        self,
        image_rgb: np.ndarray,
        *,
        prompt: str,
        negative_prompt: str,
        caption: str,
        seed: int,
        saliency_map: np.ndarray | None = None,
    ) -> np.ndarray:
        out, _ = self.generate_with_aux(
            image_rgb,
            prompt=prompt,
            negative_prompt=negative_prompt,
            caption=caption,
            seed=seed,
            saliency_map=saliency_map,
        )
        return out


def create_backend(cfg: Dict[str, Any]) -> AugmentationBackend:
    name = str(cfg.get("backend", "opencv_caption_style")).lower()

    if name in {"opencv", "opencv_caption_style"}:
        return OpenCVCaptionStyleBackend(cfg.get("opencv", {}))
    if name in {"diffusers", "diffusers_img2img"}:
        return DiffusersImg2ImgBackend(cfg.get("diffusers", {}))

    raise ValueError(
        f"Unknown generation backend '{name}'. "
        "Expected one of: opencv_caption_style, diffusers_img2img"
    )

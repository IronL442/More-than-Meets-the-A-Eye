from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_DEFAULT_STYLE_PHRASES = (
    "cinematic natural lighting",
    "soft diffused lighting",
    "high dynamic range",
    "documentary photograph style",
    "clean studio-like lighting",
    "slightly warm color grading",
    "slightly cool color grading",
)

_DEFAULT_CAMERA_PHRASES = (
    "slightly shifted viewpoint",
    "mild zoom variation",
    "small focal-length variation",
    "subtle framing change",
)

_DEFAULT_NEGATIVE_PROMPT = (
    "extra people, duplicated limbs, text overlays, logos, watermarks, "
    "cartoon style, distorted anatomy, surreal artifacts"
)


@dataclass(frozen=True)
class PromptResult:
    positive: str
    negative: str
    style_tag: str


class PromptBuilder:
    """Caption-to-prompt helper for semantics-preserving augmentations."""

    def __init__(
        self,
        *,
        style_phrases: Iterable[str] | None = None,
        camera_phrases: Iterable[str] | None = None,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    ) -> None:
        self.style_phrases = tuple(style_phrases) if style_phrases is not None else _DEFAULT_STYLE_PHRASES
        self.camera_phrases = tuple(camera_phrases) if camera_phrases is not None else _DEFAULT_CAMERA_PHRASES
        if not self.style_phrases:
            raise ValueError("PromptBuilder requires at least one style phrase.")
        if not self.camera_phrases:
            raise ValueError("PromptBuilder requires at least one camera phrase.")
        self.negative_prompt = str(negative_prompt)

    @staticmethod
    def _clean_caption(caption: str) -> str:
        text = " ".join(str(caption).strip().split())
        if text.endswith("."):
            text = text[:-1]
        return text

    def build(self, caption: str, rng: np.random.Generator) -> PromptResult:
        clean_caption = self._clean_caption(caption)
        style = str(rng.choice(self.style_phrases))
        camera = str(rng.choice(self.camera_phrases))

        positive = (
            "Photorealistic image edit of the same scene: "
            f"{clean_caption}. "
            f"Apply {style}, with {camera}. "
            "Preserve scene structure, object identities, and spatial layout."
        )

        return PromptResult(
            positive=positive,
            negative=self.negative_prompt,
            style_tag=style,
        )

"""AugSal-like data augmentation pipeline components."""

from .backends import create_backend
from .prompting import PromptBuilder
from .pseudo_label import (
    build_pseudo_label,
    compute_change_attention,
    renorm_prob,
    select_saliency_guided_attention_map,
)

__all__ = [
    "create_backend",
    "PromptBuilder",
    "build_pseudo_label",
    "compute_change_attention",
    "renorm_prob",
    "select_saliency_guided_attention_map",
]

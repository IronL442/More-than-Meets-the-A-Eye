import argparse
from typing import Iterable, List

import torch

import deepgaze_pytorch
from deepgaze_pytorch import deepgaze2e


def _format_seq(seq: torch.nn.Module) -> List[str]:
    lines: List[str] = []
    for name, module in seq.named_children():
        lines.append(f"{name}: {module.__class__.__name__}")
    return lines


def _print_block(title: str, lines: Iterable[str]) -> None:
    print(f"{title}:")
    for line in lines:
        print(f"  - {line}")


def _print_head_arch() -> None:
    sal = deepgaze2e.build_saliency_network(input_channels=2048)
    fix = deepgaze2e.build_fixation_selection_network()
    print("Head components")
    _print_block("Saliency network (per component)", _format_seq(sal))
    _print_block("Fixation selection network (per component)", _format_seq(fix))
    print("Finalizer (per component)")
    print("  - GaussianFilterNd (learn_sigma=True)")
    print("  - center_bias_weight (learnable scalar)")
    print("")


def _print_backbones() -> None:
    print("Backbones")
    for i, cfg in enumerate(deepgaze2e.BACKBONES):
        print(f"  [{i}] {cfg['type']}")
        print("      used_features:")
        for feat in cfg["used_features"]:
            print(f"        - {feat}")
        print(f"      channels: {cfg['channels']}")
    print("")


def _print_block_mapping() -> None:
    print("Config block mapping")
    print("  backbone          -> feature extractors (one per backbone)")
    print("  saliency          -> saliency_networks (per component)")
    print("  fixation_selection-> fixation_selection_networks (per component)")
    print("  finalizer         -> finalizers (per component)")
    print("  head              -> saliency + fixation_selection + finalizer")
    print("  all               -> entire model")
    print("")


def _try_instantiate() -> None:
    print("Attempting to instantiate DeepGazeIIE (may download weights)...")
    try:
        model = deepgaze_pytorch.DeepGazeIIE(pretrained=False)
        print("Model instantiated.")
        print(model)
    except Exception as exc:  # pragma: no cover - best effort inspection
        print("Failed to instantiate model (likely due to missing pretrained weights).")
        print(f"Reason: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instantiate", action="store_true", help="Try to build the full model (may download weights).")
    args = parser.parse_args()

    print("DeepGaze II-E architecture summary")
    print("----------------------------------")
    _print_backbones()
    _print_head_arch()
    _print_block_mapping()

    if args.instantiate:
        _try_instantiate()


if __name__ == "__main__":
    main()

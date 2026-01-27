import re
from typing import Dict, Iterable, List, Tuple

import torch


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _leaf_modules_with_params(
    model: torch.nn.Module,
) -> Iterable[Tuple[str, torch.nn.Module]]:
    for name, module in model.named_modules():
        if list(module.children()):
            continue
        params = list(module.parameters(recurse=False))
        if params:
            yield name, module


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    _set_requires_grad(module, requires_grad)


def apply_freeze_config(model: torch.nn.Module, cfg: Dict) -> Dict[str, int]:
    """
    Apply a freeze/unfreeze policy to a model.

    Supported modes:
      - train_all / freeze_all
      - train_last_n_modules / freeze_last_n_modules
      - train_last_n_params / freeze_last_n_params
      - train_regex / freeze_regex (patterns list of regex strings)
    """
    cfg = cfg or {}
    mode = str(cfg.get("mode", "train_all"))
    n = int(cfg.get("n", 0))
    patterns = cfg.get("patterns", [])
    if isinstance(patterns, str):
        patterns = [patterns]

    if mode == "train_all":
        _set_requires_grad(model, True)
        return count_params(model)
    if mode == "freeze_all":
        _set_requires_grad(model, False)
        return count_params(model)

    if mode in ("train_last_n_modules", "freeze_last_n_modules"):
        leaf_modules = list(_leaf_modules_with_params(model))
        n = max(0, min(n, len(leaf_modules)))
        if mode == "train_last_n_modules":
            _set_requires_grad(model, False)
            for _, module in leaf_modules[-n:]:
                _set_requires_grad(module, True)
        else:
            _set_requires_grad(model, True)
            for _, module in leaf_modules[-n:]:
                _set_requires_grad(module, False)
        return count_params(model)

    if mode in ("train_last_n_params", "freeze_last_n_params"):
        params = list(model.parameters())
        n = max(0, min(n, len(params)))
        if mode == "train_last_n_params":
            _set_requires_grad(model, False)
            for param in params[-n:]:
                param.requires_grad = True
        else:
            _set_requires_grad(model, True)
            for param in params[-n:]:
                param.requires_grad = False
        return count_params(model)

    if mode in ("train_regex", "freeze_regex"):
        if not patterns:
            raise ValueError("freeze config 'patterns' is required for regex modes.")
        regex = re.compile("|".join(patterns))
        if mode == "train_regex":
            _set_requires_grad(model, False)
            for name, param in model.named_parameters():
                if regex.search(name):
                    param.requires_grad = True
        else:
            _set_requires_grad(model, True)
            for name, param in model.named_parameters():
                if regex.search(name):
                    param.requires_grad = False
        return count_params(model)

    raise ValueError(f"Unknown freeze mode: {mode}")

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:
    import torch_directml
    _HAS_DML = True
except ImportError:
    _HAS_DML = False

import deepgaze_pytorch

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.deepgaze_iie import build_deepgaze_inputs
from saliency_bench.utils.finetune_utils import apply_freeze_config, count_params, set_requires_grad
from saliency_bench.utils.gt_from_fix import fixations_to_density
from saliency_bench.utils.image_ops import renorm_prob, to_rgb_uint8


class FolderSaliencyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths: List[str],
        root: str,
        centerbias_template: np.ndarray,
        sigma_px: int = 15,
        allow_uniform_gt: bool = False,
        gt_aggregate: str = "mean",
        gt_resize_interp: str = "bilinear",
        gt_cache_dir: Optional[str] = None,
    ):
        self.img_paths = img_paths
        self.root = root
        self.centerbias_template = centerbias_template
        self.sigma_px = sigma_px
        self.allow_uniform_gt = allow_uniform_gt
        self.gt_aggregate = gt_aggregate
        self.gt_resize_interp = gt_resize_interp
        self.gt_cache_dir = gt_cache_dir
        self.gt_dir = os.path.join(root, "gt_maps")
        self.fix_dir = os.path.join(root, "fixations")
        self.gt_paths_per_image: List[List[str]] = []
        self.stem_to_index: Dict[str, int] = {}

        missing = []
        for idx, p in enumerate(self.img_paths):
            stem = os.path.splitext(os.path.basename(p))[0]
            self.stem_to_index[stem] = idx
            gt_paths = self._find_gt_paths(stem)
            self.gt_paths_per_image.append(gt_paths)
            if not allow_uniform_gt:
                fix_path = os.path.join(self.fix_dir, stem + ".npy")
                if not (gt_paths or os.path.exists(fix_path)):
                    missing.append(stem)
        if missing:
            raise ValueError(
                "Missing gt_maps or fixations for some images. "
                "Provide labels or set data.allow_uniform_gt: true. "
                f"Example missing: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.img_paths)

    def _find_gt_paths(self, stem: str) -> List[str]:
        if not os.path.isdir(self.gt_dir):
            return []
        import glob
        exts = (".npy", ".png", ".jpg", ".jpeg", ".bmp")
        direct = os.path.join(self.gt_dir, stem + ".npy")
        if os.path.exists(direct):
            return [direct]
        paths: List[str] = []
        for ext in exts:
            pattern = os.path.join(self.gt_dir, f"*_{stem}{ext}")
            paths.extend(glob.glob(pattern))
        return sorted(paths)

    def _resize_map(self, arr: np.ndarray, H: int, W: int) -> np.ndarray:
        if arr.shape == (H, W):
            return arr.astype(np.float32)
        interp = self.gt_resize_interp.lower()
        if interp == "bilinear":
            interp_flag = cv2.INTER_LINEAR
        elif interp == "bicubic":
            interp_flag = cv2.INTER_CUBIC
        else:
            raise ValueError(f"Unknown gt_resize_interp: {self.gt_resize_interp}")
        resized = cv2.resize(arr.astype(np.float32), (W, H), interpolation=interp_flag)
        return resized

    def _load_gt_map(self, stem: str, H: int, W: int) -> Optional[np.ndarray]:
        idx = self.stem_to_index.get(stem, -1)
        gt_paths = self.gt_paths_per_image[idx] if idx >= 0 else self._find_gt_paths(stem)
        if not gt_paths:
            return None

        if self.gt_aggregate == "mean" and self.gt_cache_dir:
            cached_path = os.path.join(self.gt_cache_dir, stem + ".npy")
            if not os.path.exists(cached_path):
                raise FileNotFoundError(
                    f"Missing cached mean map for {stem} in {self.gt_cache_dir}. "
                    "Run scripts/precompute_gt_mean.py first."
                )
            cached = np.load(cached_path).astype(np.float32)
            if cached.shape != (H, W):
                cached = self._resize_map(cached, H, W)
                cached = np.clip(cached, 0.0, None)
                cached = renorm_prob(cached)
            return cached

        maps = []
        for path in gt_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".npy":
                arr = np.load(path).astype(np.float32)
            else:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"Failed to read GT map: {path}")
                arr = img.astype(np.float32)
            arr = self._resize_map(arr, H, W)
            arr = np.clip(arr, 0.0, None)
            maps.append(renorm_prob(arr))
        if self.gt_aggregate == "random":
            return maps[np.random.randint(0, len(maps))]
        if self.gt_aggregate == "median":
            return np.median(np.stack(maps, axis=0), axis=0).astype(np.float32)
        if self.gt_aggregate == "mean":
            mean_map = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
            mean_map = renorm_prob(mean_map)
            return mean_map
        raise ValueError(f"Unknown gt_aggregate: {self.gt_aggregate}")

    def __getitem__(self, idx: int) -> Dict[str, Optional[torch.Tensor]]:
        path = self.img_paths[idx]
        stem = os.path.splitext(os.path.basename(path))[0]

        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = to_rgb_uint8(img)
        H, W = img.shape[:2]

        gt = self._load_gt_map(stem, H, W)

        fix: Optional[np.ndarray] = None
        fix_path = os.path.join(self.fix_dir, stem + ".npy")
        if os.path.exists(fix_path):
            fix = np.load(fix_path).astype(np.uint8)
            if fix.shape != (H, W):
                fix = cv2.resize(fix.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            if gt is None:
                try:
                    gt = fixations_to_density(fix, sigma_px=self.sigma_px)
                except Exception:
                    gt = None

        if gt is None:
            gt = np.full((H, W), 1.0 / (H * W), dtype=np.float32)

        gt = renorm_prob(gt)
        image_tensor, centerbias_tensor = build_deepgaze_inputs(
            img,
            self.centerbias_template,
            device="cpu",
            add_batch_dim=False,
        )

        gt_tensor = torch.from_numpy(gt).float()
        fix_tensor = torch.from_numpy(fix).float() if fix is not None else torch.zeros_like(gt_tensor)
        return {
            "image": image_tensor,
            "centerbias": centerbias_tensor,
            "gt_map": gt_tensor,
            "fixations": fix_tensor,
        }


def _select_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if _HAS_DML:
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_feature_train_mode(model: torch.nn.Module, mode: bool) -> None:
    for module in model.modules():
        if module.__class__.__name__ == "FeatureExtractor":
            module.train(mode=mode)


def _deepgaze_iie_blocks(model: torch.nn.Module) -> Dict[str, List[torch.nn.Module]]:
    blocks: Dict[str, List[torch.nn.Module]] = {
        "backbone": [],
        "saliency": [],
        "fixation_selection": [],
        "finalizer": [],
        "head": [],
        "all": [model],
    }
    if hasattr(model, "models"):
        for mixture in model.models:
            if hasattr(mixture, "features"):
                blocks["backbone"].append(mixture.features)
            if hasattr(mixture, "saliency_networks"):
                blocks["saliency"].append(mixture.saliency_networks)
            if hasattr(mixture, "fixation_selection_networks"):
                blocks["fixation_selection"].append(mixture.fixation_selection_networks)
            if hasattr(mixture, "finalizers"):
                blocks["finalizer"].append(mixture.finalizers)
    blocks["head"] = blocks["saliency"] + blocks["fixation_selection"] + blocks["finalizer"]
    return blocks


def _apply_named_blocks(model: torch.nn.Module, cfg: Dict) -> Dict[str, int]:
    mode = str(cfg.get("mode", "train_blocks"))
    blocks = cfg.get("blocks", [])
    if isinstance(blocks, str):
        blocks = [blocks]
    if not blocks:
        raise ValueError("freeze config 'blocks' is required for named block modes.")

    block_map = _deepgaze_iie_blocks(model)
    modules: List[torch.nn.Module] = []
    for name in blocks:
        if name not in block_map:
            raise ValueError(f"Unknown DeepGaze block: {name}")
        modules.extend(block_map[name])

    if mode == "train_blocks":
        set_requires_grad(model, False)
        for module in modules:
            set_requires_grad(module, True)

        backbone_last_n = int(cfg.get("backbone_last_n_modules", 0))
        if backbone_last_n > 0 and hasattr(model, "models"):
            for mixture in model.models:
                if hasattr(mixture, "features"):
                    apply_freeze_config(
                        mixture.features,
                        {"mode": "train_last_n_modules", "n": backbone_last_n},
                    )
        return count_params(model)

    if mode == "freeze_blocks":
        set_requires_grad(model, True)
        for module in modules:
            set_requires_grad(module, False)
        return count_params(model)

    raise ValueError(f"Unknown named-block mode: {mode}")

def _apply_freeze(model: torch.nn.Module, cfg: Dict) -> Dict[str, int]:
    scope = str(cfg.get("scope", "all"))
    mode = str(cfg.get("mode", ""))
    if mode in ("train_blocks", "freeze_blocks"):
        return _apply_named_blocks(model, cfg)
    if scope == "all":
        apply_freeze_config(model, cfg)
        return count_params(model)
    if scope == "backbone":
        if hasattr(model, "models"):
            for mixture in model.models:
                if hasattr(mixture, "features"):
                    apply_freeze_config(mixture.features, cfg)
        else:
            apply_freeze_config(model, cfg)
        return count_params(model)
    raise ValueError(f"Unknown freeze scope: {scope}")


def _loss_from_batch(
    log_pred: torch.Tensor,
    gt_map: torch.Tensor,
    fixations: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    log_pred = log_pred - log_pred.logsumexp(dim=(1, 2), keepdim=True)
    if loss_type == "kl":
        gt_map = gt_map / (gt_map.sum(dim=(1, 2), keepdim=True) + 1e-8)
        return F.kl_div(log_pred, gt_map, reduction="batchmean")
    if loss_type == "nll_fixations":
        if float(fixations.sum().detach().cpu().numpy()) == 0.0:
            raise ValueError("Fixations are required for loss_type: nll_fixations")
        fix = fixations / (fixations.sum(dim=(1, 2), keepdim=True) + 1e-8)
        return -(log_pred * fix).sum(dim=(1, 2)).mean()
    raise ValueError(f"Unknown loss type: {loss_type}")


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_type: str,
    train: bool,
    train_features: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    if train:
        model.train()
        _set_feature_train_mode(model, train_features)
    else:
        model.eval()

    losses: List[float] = []
    for batch in loader:
        image = batch["image"].to(device)
        centerbias = batch["centerbias"].to(device)
        gt_map = batch["gt_map"].to(device)
        fixations = batch["fixations"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        log_pred = model(image, centerbias)  # [B,1,H,W]
        log_pred = log_pred[:, 0]
        loss = _loss_from_batch(log_pred, gt_map, fixations, loss_type)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu().numpy()))

    return float(np.mean(losses)) if losses else float("nan")


def _load_image_paths(root: str) -> List[str]:
    import glob
    img_dir = os.path.join(root, "images")
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(sorted([p for p in glob.glob(os.path.join(img_dir, ext))]))
    return paths


def _split_paths(paths: List[str], train_split: float, seed: int, shuffle: bool) -> Tuple[List[str], List[str]]:
    paths = list(paths)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(paths)
    split_idx = int(len(paths) * train_split)
    return paths[:split_idx], paths[split_idx:]


def _write_yaml(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run(cfg_path: str, print_counts: bool = False) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    device = torch.device("cpu") if print_counts else _select_device(model_cfg.get("device"))
    pretrained = bool(model_cfg.get("pretrained", True))

    model = deepgaze_pytorch.DeepGazeIIE(pretrained=pretrained).to(device)

    loss_type = str(cfg.get("training", {}).get("loss", "kl"))
    stages = cfg.get("training", {}).get("stages", [])
    if not stages:
        raise ValueError("No training stages provided in config.training.stages")

    if print_counts:
        print("Trainable parameter counts per stage")
        for stage in stages:
            stage_name = str(stage.get("name", "stage"))
            freeze_cfg = stage.get("freeze", {})
            param_counts = _apply_freeze(model, freeze_cfg)
            print(
                f"{stage_name}: trainable={param_counts['trainable']} total={param_counts['total']}"
            )
        return

    centerbias_path = model_cfg.get("centerbias_path", "data/centerbias/centerbias_mit1003.npy")
    use_uniform_centerbias = bool(model_cfg.get("use_uniform_centerbias", False))
    if use_uniform_centerbias:
        centerbias_template = np.zeros((1024, 1024), dtype=np.float32)
    else:
        if not os.path.exists(centerbias_path):
            raise FileNotFoundError(
                f"Centerbias file not found at {centerbias_path}. "
                "Download it from the DeepGaze repo releases."
            )
        centerbias_template = np.load(centerbias_path).astype(np.float32)

    output_dir = cfg.get("output_dir", "outputs/finetune/deepgaze_iie")
    os.makedirs(output_dir, exist_ok=True)
    _write_yaml(os.path.join(output_dir, "config.yaml"), cfg)

    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "data/img_bin")
    train_split = float(data_cfg.get("train_split", 0.8))
    seed = int(cfg.get("seed", 1337))
    shuffle = bool(data_cfg.get("shuffle", True))
    sigma_px = int(data_cfg.get("sigma_px", 15))
    allow_uniform_gt = bool(data_cfg.get("allow_uniform_gt", False))
    gt_aggregate = str(data_cfg.get("gt_aggregate", "mean"))
    gt_resize_interp = str(data_cfg.get("gt_resize_interp", "bilinear"))
    gt_cache_dir = data_cfg.get("gt_cache_dir", None)
    batch_size = int(data_cfg.get("batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 0))

    if batch_size != 1:
        raise ValueError("DeepGaze IIE fine-tuning expects batch_size=1 unless you add a custom collate/resize.")

    img_paths = _load_image_paths(root)
    if not img_paths:
        raise FileNotFoundError(f"No images found under {root}/images")

    train_paths, val_paths = _split_paths(img_paths, train_split, seed, shuffle)

    train_ds = FolderSaliencyDataset(
        train_paths,
        root=root,
        centerbias_template=centerbias_template,
        sigma_px=sigma_px,
        allow_uniform_gt=allow_uniform_gt,
        gt_aggregate=gt_aggregate,
        gt_resize_interp=gt_resize_interp,
        gt_cache_dir=gt_cache_dir,
    )
    val_ds = FolderSaliencyDataset(
        val_paths,
        root=root,
        centerbias_template=centerbias_template,
        sigma_px=sigma_px,
        allow_uniform_gt=allow_uniform_gt,
        gt_aggregate=gt_aggregate,
        gt_resize_interp=gt_resize_interp,
        gt_cache_dir=gt_cache_dir,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    log_path = os.path.join(output_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("stage,epoch,train_loss,val_loss,trainable_params,total_params\n")

    for stage in stages:
        stage_name = str(stage.get("name", "stage"))
        stage_dir = os.path.join(output_dir, stage_name)
        os.makedirs(stage_dir, exist_ok=True)

        freeze_cfg = stage.get("freeze", {})
        param_counts = _apply_freeze(model, freeze_cfg)

        lr = float(stage.get("lr", 1e-4))
        weight_decay = float(stage.get("weight_decay", 0.0))
        epochs = int(stage.get("epochs", 1))
        train_features = bool(stage.get("train_features", False))

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

        for epoch in range(1, epochs + 1):
            train_loss = _run_epoch(
                model,
                train_loader,
                device,
                loss_type=loss_type,
                train=True,
                train_features=train_features,
                optimizer=optimizer,
            )
            val_loss = _run_epoch(
                model,
                val_loader,
                device,
                loss_type=loss_type,
                train=False,
                train_features=train_features,
            )

            ckpt_path = os.path.join(stage_dir, f"epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{stage_name},{epoch},{train_loss:.6f},{val_loss:.6f},"
                    f"{param_counts['trainable']},{param_counts['total']}\n"
                )

    final_path = os.path.join(output_dir, "final.pth")
    torch.save(model.state_dict(), final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--print_counts", action="store_true")
    args = parser.parse_args()
    run(args.config, print_counts=args.print_counts)

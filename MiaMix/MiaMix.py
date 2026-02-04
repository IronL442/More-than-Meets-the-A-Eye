import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import List, Tuple, Union
import os
from glob import glob
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

Tensor = torch.Tensor


# -------------------------------------------------------
# Utility: Dirichlet sampling (λ₁…λₖ₊₁)
# -------------------------------------------------------
def sample_dirichlet(alpha: float, k: int) -> np.ndarray:
    """
    Samples λ₁…λₖ₊₁ from Dir(α, …, α).
    """
    alpha_vec = np.full(k + 1, alpha, dtype=np.float32)
    return np.random.dirichlet(alpha_vec)


# -------------------------------------------------------
# MiAMix Module (Algorithm-faithful)
# -------------------------------------------------------
class MiAMix(nn.Module):
    """
    Algorithm 1: Multi-stage Augmented Mixup (MiaMix)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        k_max: int = 3,
        prob_self: float = 0.1,
        methods: List[str] = ("mixup", "cutmix", "agmix"),
        method_weights: List[float] = None,
        device: str = "cuda",
    ):
        super().__init__()

        self.alpha = alpha
        self.k_max = k_max
        self.prob_self = prob_self
        self.methods = list(methods)
        self.device = device

        if method_weights is None:
            self.method_weights = np.ones(len(methods)) / len(methods)
        else:
            self.method_weights = np.array(method_weights) / np.sum(method_weights)

    # ---------------------------------------------------
    # Mask augmentation (Algorithm: Apply mask augmentation)
    # ---------------------------------------------------
    def augment_mask(self, mask: Tensor) -> Tensor:
        angle = np.random.uniform(-15, 15)

        mask_b = mask.unsqueeze(0)
        theta = torch.tensor(
            [[
                [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0.0],
                [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0.0],
            ]],
            device=self.device,
            dtype=torch.float32,
        )

        grid = F.affine_grid(theta, mask_b.size(), align_corners=False)
        mask = F.grid_sample(mask_b, grid, align_corners=False).squeeze(0)

        k = int(np.random.choice([3, 5, 7]))
        sigma = np.random.uniform(0.3, 1.2)
        mask = TF.gaussian_blur(mask, [k, k], [sigma, sigma])

        return mask.clamp(0, 1)

    # ---------------------------------------------------
    # Individual mask generators (m_j(λ_j))
    # ---------------------------------------------------
    def cutmix_mask(self, H: int, W: int, lam: float) -> Tensor:
        cut_w = int(W * np.sqrt(1 - lam))
        cut_h = int(H * np.sqrt(1 - lam))
        cx, cy = np.random.randint(W), np.random.randint(H)

        mask = torch.ones(1, H, W, device=self.device)
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, H)
        mask[:, y1:y2, x1:x2] = 0.0
        return mask

    def agmix_mask(self, H: int, W: int, lam: float) -> Tensor:
        mu_x, mu_y = np.random.uniform(0, W), np.random.uniform(0, H)
        sigma = np.sqrt(1 - lam) * max(H, W) / 4

        y, x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )

        g = torch.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))
        return (g > 0.5).float().unsqueeze(0)

    # ---------------------------------------------------
    # Forward = Algorithm main loop
    # ---------------------------------------------------
    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x : (N, C, H, W)
        y : (N, C_y, H, W)
        """

        N, _, H, W = x.shape
        _, C_y, _, _ = y.shape

        out_x, out_y = [], []
        self_mix_flags = []

        for i in range(N):

            # ------------------------------------------------
            # Sample mixing data point (x_t, y_t)
            # ------------------------------------------------
            if np.random.rand() < self.prob_self:
                j = i
                self_mix=True
            else:
                j = np.random.randint(0, N)
                self_mix= (j==i)

            x_i, y_i = x[i], y[i]
            x_t, y_t = x[j], y[j]

            # ------------------------------------------------
            # Sample number of mixing layers k
            # ------------------------------------------------
            k = np.random.randint(1, self.k_max + 1)

            # ------------------------------------------------
            # Sample λ₁…λₖ₊₁ from Dirichlet
            # ------------------------------------------------
            lambdas = sample_dirichlet(self.alpha, k)

            # ------------------------------------------------
            # Sample mixing methods m₁…mₖ
            # ------------------------------------------------
            methods = np.random.choice(
                self.methods, size=k, p=self.method_weights
            )

            # ------------------------------------------------
            # Generate masks m_j(λ_j)
            # ------------------------------------------------
            masks = []
            for j_stage in range(k):
                lam_j = lambdas[j_stage]
                m = methods[j_stage]

                if m == "mixup":
                    mask = torch.full((1, H, W), lam_j, device=self.device)
                elif m == "cutmix":
                    mask = self.cutmix_mask(H, W, lam_j)
                elif m == "agmix":
                    mask = self.agmix_mask(H, W, lam_j)
                else:
                    raise ValueError(m)

                mask = self.augment_mask(mask)
                masks.append(mask)

            # ------------------------------------------------
            # Merge all k masks
            # ------------------------------------------------
            mask_merged = torch.stack(masks).mean(dim=0)
            lambda_merged = mask_merged.mean().item()

            # ------------------------------------------------
            # Apply mask to input and label
            # ------------------------------------------------
            mask_x = mask_merged.repeat(x_i.shape[0], 1, 1)
            mask_y = mask_merged.repeat(C_y, 1, 1)

            x_mix = mask_x * x_i + (1 - mask_x) * x_t
            y_mix = lambda_merged * y_i + (1 - lambda_merged) * y_t

            out_x.append(x_mix)
            out_y.append(y_mix)
            self_mix_flags.append(self_mix)

        return torch.stack(out_x), torch.stack(out_y), self_mix_flags
    

if __name__ == "__main__":

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------
    IMAGE_DIR = "data/seminar_data/images"
    HEATMAP_DIR = "data/seminar_data/heatmaps"

    OUTPUT_IMAGE_DIR = "MiaMix/augmented_images/images"
    OUTPUT_LABEL_DIR = "MiaMix/augmented_images/labels"

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    IMAGE_SIZE = 224
    BATCH_SIZE = 4
    NUM_CLASSES = 1

    # MiaMix parameters (Algorithm Line 2)
    ALPHA = 0.5
    K_MAX = 3
    PROB_SELF = 0.5

    METHODS = ["mixup", "cutmix", "agmix"]
    # W: Sampling weights for methods (Algorithm Line 2)
    METHOD_WEIGHTS = [0.33, 0.33, 0.34]

    print(f"Using device: {DEVICE}")

    # --------------------------------------------------
    # Transforms
    # --------------------------------------------------
    image_tf = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])

    heatmap_tf = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])

    # --------------------------------------------------
    # Load image paths
    # --------------------------------------------------
    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))

    if len(image_paths) == 0:
        raise RuntimeError("No images found.")

    # Initialize MiaMix ONCE
    miamix = MiAMix(
        alpha=ALPHA,
        k_max=K_MAX,
        prob_self=PROB_SELF,
        methods=METHODS,
        method_weights=METHOD_WEIGHTS,
        device=DEVICE,
    ).to(DEVICE)

    # --------------------------------------------------
    # Process ALL images in batches
    # --------------------------------------------------
    for batch_start in range(0, len(image_paths), BATCH_SIZE):

        batch_paths = image_paths[batch_start: batch_start + BATCH_SIZE]

        imgs, hmaps, names = [], [], []

        for path in batch_paths:
            base = os.path.splitext(os.path.basename(path))[0]

            # Find ALL matching heatmaps for this image core (e.g., P01_base.png, P02_base.png)
            # This prepares the consensus label y_i as required by Algorithm Line 1
            heatmap_matches = glob(os.path.join(HEATMAP_DIR, f"*{base}*.png"))
            
            if not heatmap_matches:
                print(f"Skipping {base}: no heatmap found")
                continue

            # Load and transform the base image
            img = Image.open(path).convert("RGB")
            imgs.append(image_tf(img))

            # --------------------------------------------------
            # Heatmap Averaging Logic (Consensus Ground Truth)
            # --------------------------------------------------
            person_hmaps = []
            for h_match in heatmap_matches:
                # Load each annotator's heatmap
                h_img = Image.open(h_match).convert("L")
                person_hmaps.append(heatmap_tf(h_img)) # Shape: (1, H, W)
            
            # Stack all persons and compute the mean to get consensus y_i
            # 
            stacked_hmaps = torch.stack(person_hmaps) # (Num_Persons, 1, H, W)
            avg_hmap = torch.mean(stacked_hmaps, dim=0) # (1, H, W)
            
            # Replicate to match channel dimensions and append to batch
            hmaps.append(avg_hmap.repeat(NUM_CLASSES, 1, 1))
            names.append(base)

        # Skip if no valid samples were found in this batch
        if len(imgs) == 0:
            continue

        x = torch.stack(imgs).to(DEVICE)
        y = torch.stack(hmaps).to(DEVICE)

        # --------------------------------------------------
        # Apply MiaMix (Algorithm Lines 5-16)
        # --------------------------------------------------
        with torch.no_grad():
            x_mix, y_mix, self_mix_flags = miamix(x, y)

        # --------------------------------------------------
        # Save augmented results (Algorithm Line 17)
        # --------------------------------------------------
        for i in range(x_mix.size(0)):
            # Save the mixed sample x_tilde
            save_image(
                x_mix[i],
                os.path.join(OUTPUT_IMAGE_DIR, f"{names[i]}_AUG.jpg")
            )
            # Save the mixed label y_tilde (first channel)
            save_image(
                y_mix[i][0],
                os.path.join(OUTPUT_LABEL_DIR, f"{names[i]}_AUG.png")
            )

            mix_type = "SELF-MIX" if self_mix_flags[i] else "CROSS-MIX"
            print(f"Saved augmented sample for {names[i]} | Mix: {mix_type} | Annotators averaged: {len(heatmap_matches)}")

    print("\nMiaMix augmentation with multi-person averaging finished successfully.")
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def visualize_maps(samples, titles):
    n = len(samples)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axs = [axs]
    for i in range(n):
        axs[i].imshow(samples[i], cmap='hot')
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()


def compute_entropy(map_arr):
    epsilon = 1e-12
    normed = map_arr / (np.sum(map_arr) + epsilon)
    return -np.sum(normed * np.log(normed + epsilon))


def normalize_map(map_arr, mode="max"):
    arr = np.asarray(map_arr, dtype=np.float32)
    if mode == "max":
        vmax = float(arr.max())
        if vmax > 0:
            arr = arr / vmax
    elif mode == "sum":
        s = float(arr.sum())
        if s > 0:
            arr = arr / s
            vmax = float(arr.max())
            if vmax > 0:
                arr = arr / vmax
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return np.clip(arr, 0.0, 1.0)


def _resolve_selected_files(npy_files, image_id):
    if image_id is None:
        return npy_files

    image_id = image_id.strip()
    if not image_id:
        raise ValueError("--image_id cannot be empty")

    target_fname = image_id if image_id.endswith(".npy") else f"{image_id}.npy"
    if target_fname not in npy_files:
        raise FileNotFoundError(
            f"Map for image_id '{image_id}' not found. "
            f"Expected file: {target_fname}"
        )
    return [target_fname]


def main(
    gt_cache_dir,
    num_samples,
    image_id=None,
    save_png=False,
    out_dir="outputs/gt_heatmaps",
    all_files=False,
    colormap="hot",
    normalize="max",
    no_show=False,
):
    if not os.path.isdir(gt_cache_dir):
        raise FileNotFoundError(f"{gt_cache_dir} does not exist")

    npy_files = sorted([f for f in os.listdir(gt_cache_dir) if f.endswith(".npy")])
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No .npy files found in {gt_cache_dir}")

    selected = _resolve_selected_files(npy_files, image_id)
    if image_id is not None:
        print(f"Found {len(npy_files)} maps. Displaying selected image_id: {selected[0][:-4]}")
        chosen = selected
    elif all_files:
        print(f"Found {len(npy_files)} maps. Processing all files...")
        chosen = selected
    else:
        print(f"Found {len(npy_files)} maps. Displaying {num_samples}...")
        chosen = selected[:num_samples]

    if save_png:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving PNG heatmaps to: {out_dir}")

    maps = []
    titles = []

    for fname in chosen:
        path = os.path.join(gt_cache_dir, fname)
        arr = np.load(path).astype(np.float32)
        entropy = compute_entropy(arr)
        if save_png:
            out_path = os.path.join(out_dir, f"{fname[:-4]}.png")
            norm = normalize_map(arr, mode=normalize)
            plt.imsave(out_path, norm, cmap=colormap, vmin=0.0, vmax=1.0)
        if not no_show:
            maps.append(arr)
            titles.append(f"{fname[:-4]}\nEntropy: {entropy:.3f}")

    if not no_show:
        visualize_maps(maps, titles)
    elif not save_png:
        print("Nothing to do: --no_show was set and --save_png was not set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize precomputed saliency maps (.npy)")
    parser.add_argument("--gt_cache_dir", type=str, required=True, help="Path to GT cache dir")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of maps to visualize")
    parser.add_argument(
        "--image_id",
        type=str,
        default=None,
        help="Specific image id or filename (.npy) to visualize. Overrides list selection.",
    )
    parser.add_argument("--save_png", action="store_true", help="Save each selected heatmap as PNG")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/gt_heatmaps",
        help="Output directory for saved PNGs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_files",
        help="Process all .npy files in --gt_cache_dir (ignores --num_samples)",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="hot",
        help="Matplotlib colormap for saved PNGs (e.g. hot, jet, viridis)",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="max",
        choices=["max", "sum", "none"],
        help="Normalization for saved PNGs",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open matplotlib window (useful with --save_png)",
    )
    args = parser.parse_args()

    main(
        args.gt_cache_dir,
        args.num_samples,
        image_id=args.image_id,
        save_png=args.save_png,
        out_dir=args.out_dir,
        all_files=args.all_files,
        colormap=args.colormap,
        normalize=args.normalize,
        no_show=args.no_show,
    )

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


def main(gt_cache_dir, num_samples, image_id=None):
    if not os.path.isdir(gt_cache_dir):
        raise FileNotFoundError(f"{gt_cache_dir} does not exist")

    npy_files = sorted([f for f in os.listdir(gt_cache_dir) if f.endswith(".npy")])
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No .npy files found in {gt_cache_dir}")

    selected = _resolve_selected_files(npy_files, image_id)
    if image_id is not None:
        print(f"Found {len(npy_files)} maps. Displaying selected image_id: {selected[0][:-4]}")
    else:
        print(f"Found {len(npy_files)} maps. Displaying {num_samples}...")

    maps = []
    titles = []

    for fname in selected[:num_samples]:
        path = os.path.join(gt_cache_dir, fname)
        arr = np.load(path).astype(np.float32)
        entropy = compute_entropy(arr)
        maps.append(arr)
        titles.append(f"{fname[:-4]}\nEntropy: {entropy:.3f}")

    visualize_maps(maps, titles)


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
    args = parser.parse_args()

    main(args.gt_cache_dir, args.num_samples, image_id=args.image_id)

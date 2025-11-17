import cv2
import numpy as np

from datasets.folder_dataset import FolderDataset


def test_folder_dataset_iter(tmp_path):
    d = tmp_path / "ds"
    (d / "images").mkdir(parents=True)
    (d / "gt_maps").mkdir()
    (d / "fixations").mkdir()
    img = np.full((20, 30, 3), 255, np.uint8)
    cv2.imwrite(str(d / "images/000.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    gt = np.ones((20, 30), np.float32)
    gt /= gt.sum()
    np.save(d / "gt_maps/000.npy", gt)
    fix = np.zeros((20, 30), np.uint8)
    fix[10, 15] = 1
    np.save(d / "fixations/000.npy", fix)

    ds = FolderDataset(root=str(d))
    sample = next(iter(ds))
    assert set(sample.keys()) >= {"image_id", "image", "gt_map", "fixations"}
    assert sample["image"].shape == (20, 30, 3)
    assert sample["gt_map"].shape == (20, 30)


import numpy as np

from metrics.metrics import cc, emd_wasserstein, kl_div


def test_metrics_shapes_and_ranges():
    H, W = 64, 64
    gt = np.zeros((H, W), np.float32)
    gt[H // 2, W // 2] = 1.0
    gt /= gt.sum()
    pred = gt.copy()

    assert np.isfinite(cc(pred, gt))
    assert np.isfinite(kl_div(pred, gt))
    assert np.isfinite(emd_wasserstein(pred, gt))
    assert cc(pred, gt) > 0.9
    assert kl_div(pred, gt) < 1e-4
    assert emd_wasserstein(pred, gt) < 1e-4


def test_uniform_pred_is_worse_than_matching_pred():
    H, W = 32, 32
    gt = np.zeros((H, W), np.float32)
    gt[H // 2, W // 2] = 1.0
    gt /= gt.sum()
    pred_match = gt.copy()
    pred_uniform = np.ones((H, W), np.float32) / (H * W)

    assert kl_div(pred_match, gt) < kl_div(pred_uniform, gt)
    assert cc(pred_match, gt) > cc(pred_uniform, gt)
    assert emd_wasserstein(pred_match, gt) < emd_wasserstein(pred_uniform, gt)

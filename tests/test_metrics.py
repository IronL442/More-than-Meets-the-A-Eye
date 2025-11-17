import numpy as np

from metrics.metrics import auc_judd, cc, kl_div, nss, sim_score


def test_metrics_shapes_and_ranges():
    H, W = 64, 64
    gt = np.zeros((H, W), np.float32)
    gt[H // 2, W // 2] = 1.0
    gt /= gt.sum()
    pred = gt.copy()
    fix = np.zeros((H, W), np.uint8)
    fix[H // 2, W // 2] = 1

    assert np.isfinite(cc(pred, gt))
    assert np.isfinite(kl_div(pred, gt))
    assert np.isfinite(sim_score(pred, gt))
    assert np.isfinite(auc_judd(pred, fix))
    assert np.isfinite(nss(pred, fix))
    assert sim_score(pred, gt) > 0.9
    assert nss(pred, fix) > 0.5


def test_uniform_pred_and_empty_fixations():
    H, W = 32, 32
    pred = np.ones((H, W), np.float32) / (H * W)
    gt = np.ones((H, W), np.float32) / (H * W)
    fix = np.zeros((H, W), np.uint8)
    assert np.isfinite(cc(pred, gt))
    assert np.isfinite(kl_div(pred, gt))
    assert np.isfinite(sim_score(pred, gt))
    assert np.isnan(auc_judd(pred, fix))
    assert np.isnan(nss(pred, fix))


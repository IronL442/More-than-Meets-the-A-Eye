import numpy as np

_TRAPZ = getattr(np, "trapezoid", np.trapz)


from saliency_bench.utils.image_ops import renorm_prob, zscore


def _fixation_coords(fix_bin: np.ndarray):
    ys, xs = np.nonzero(fix_bin > 0)
    return np.stack([ys, xs], axis=1) if len(ys) else np.zeros((0, 2), dtype=int)


def auc_judd(pred: np.ndarray, fix_bin: np.ndarray, num_thresh: int = 100) -> float:
    pred = (pred - pred.min()) / (np.ptp(pred) + 1e-8)
    yx = _fixation_coords(fix_bin)
    if yx.shape[0] == 0:
        return float("nan")
    fix_scores = pred[yx[:, 0], yx[:, 1]]
    thresholds = np.linspace(0, 1, num_thresh)
    tp = np.array([(fix_scores >= t).mean() for t in thresholds])
    fp = np.array([(pred >= t).mean() for t in thresholds])
    return float(_TRAPZ(tp, fp))


def nss(pred: np.ndarray, fix_bin: np.ndarray) -> float:
    yx = _fixation_coords(fix_bin)
    if yx.shape[0] == 0:
        return float("nan")
    z = zscore(pred)
    return float(z[yx[:, 0], yx[:, 1]].mean())


def cc(pred: np.ndarray, gt_map: np.ndarray) -> float:
    p = pred - pred.mean()
    g = gt_map - gt_map.mean()
    denom = p.std() * g.std() + 1e-8
    return float((p * g).mean() / denom)


def kl_div(pred: np.ndarray, gt_map: np.ndarray) -> float:
    p = renorm_prob(pred)
    g = renorm_prob(gt_map)
    return float((g * (np.log((g + 1e-8) / (p + 1e-8)))).sum())


def sim_score(pred: np.ndarray, gt_map: np.ndarray) -> float:
    p = renorm_prob(pred)
    g = renorm_prob(gt_map)
    return float(np.minimum(p, g).sum())


def sauc(
    pred: np.ndarray, fix_bin: np.ndarray, nonfix_bin: np.ndarray, num_thresh: int = 100
) -> float:
    pred = (pred - pred.min()) / (np.ptp(pred) + 1e-8)
    yx_fix = _fixation_coords(fix_bin)
    yx_non = _fixation_coords(nonfix_bin)
    if yx_fix.shape[0] == 0 or yx_non.shape[0] == 0:
        return float("nan")
    fix_scores = pred[yx_fix[:, 0], yx_fix[:, 1]]
    non_scores = pred[yx_non[:, 0], yx_non[:, 1]]
    thresholds = np.linspace(0, 1, num_thresh)
    tp = np.array([(fix_scores >= t).mean() for t in thresholds])
    fp = np.array([(non_scores >= t).mean() for t in thresholds])
    return float(_TRAPZ(tp, fp))

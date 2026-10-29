import numpy as np
from scipy import stats


def wilcoxon_p(a, b):
    return stats.wilcoxon(a, b, alternative="two-sided").pvalue


def corrected_ttest_p(a, b, test_fraction):
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0 if d.mean() != 0 else 1.0
    t = d.mean() / np.sqrt((1 / k + test_fraction / (1 - test_fraction)) * var)
    return 2 * stats.t.sf(abs(t), k - 1)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def cliffs_delta(a, b):
    diff = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    return ((diff > 0).sum() - (diff < 0).sum()) / diff.size


def cliffs_magnitude(delta):
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"

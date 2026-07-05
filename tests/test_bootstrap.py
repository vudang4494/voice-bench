import numpy as np
from voicebench.metrics.bootstrap import bca_ci


def test_point_is_mean():
    vals = list(range(1, 11))  # mean = 5.5
    r = bca_ci(vals, n_resamples=2000)
    assert abs(r["point"] - 5.5) < 1e-9


def test_ci_brackets_point():
    rng = np.random.default_rng(0)
    vals = rng.normal(10, 2, size=200).tolist()
    r = bca_ci(vals, n_resamples=2000)
    assert r["low"] < r["point"] < r["high"]
    assert not r["degenerate"]


def test_degenerate_zero_variance():
    r = bca_ci([3.0, 3.0, 3.0])
    assert r["degenerate"] is True
    assert r["low"] == r["high"] == r["point"] == 3.0


def test_reproducible_seed():
    vals = [1.0, 2.0, 3.5, 4.2, 5.9, 2.1, 3.3]
    a = bca_ci(vals, seed=123, n_resamples=1000)
    b = bca_ci(vals, seed=123, n_resamples=1000)
    assert a["low"] == b["low"] and a["high"] == b["high"]


# --- T1 fix: jackknife suy biến không được trả NaN với degenerate=False ---

def test_degenerate_jackknife_median_no_nan():
    # data không suy biến (ptp>0) nhưng jackknife của median suy biến
    r = bca_ci([0.0] * 6 + [1.0], statistic=np.median, n_resamples=500)
    assert r["degenerate"] is True
    assert np.isfinite(r["low"]) and np.isfinite(r["high"])
    assert r["low"] == r["high"] == r["point"] == 0.0

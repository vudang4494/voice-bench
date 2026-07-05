"""Bootstrap BCa CI qua scipy.stats.bootstrap (method='BCa', n=10000).

Edge case: BCa cần acceleration -> khi phương sai=0 hoặc n<2, scipy lỗi/NaN.
Ngoài ra jackknife của STATISTIC có thể suy biến dù data không suy biến
(vd median trên data tied: [0]*6+[1]) -> scipy trả CI NaN. Cả hai trường hợp
đều fallback: CI suy biến = point, degenerate=True. KHÔNG BAO GIỜ trả NaN
với degenerate=False."""
from __future__ import annotations

from typing import Sequence, Callable
import warnings
import numpy as np
from scipy.stats import bootstrap


def bca_ci(
    values: Sequence[float],
    statistic: Callable[..., float] = np.mean,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    a = np.asarray([v for v in values if v == v], dtype=float)
    point = float(statistic(a)) if a.size else float("nan")
    if a.size < 2 or np.ptp(a) == 0:
        return {"point": point, "low": point, "high": point,
                "cl": confidence_level, "n": int(a.size), "degenerate": True}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = bootstrap(
            (a,), statistic, n_resamples=n_resamples,
            confidence_level=confidence_level, method="BCa",
            rng=np.random.default_rng(seed), vectorized=True,
        )
    low = float(res.confidence_interval.low)
    high = float(res.confidence_interval.high)
    if not (np.isfinite(low) and np.isfinite(high)):
        # Jackknife của statistic suy biến (acceleration 0/0) -> scipy trả NaN.
        return {"point": point, "low": point, "high": point,
                "cl": confidence_level, "n": int(a.size), "degenerate": True}
    return {
        "point": point,
        "low": low,
        "high": high,
        "cl": confidence_level,
        "n": int(a.size),
        "degenerate": False,
    }

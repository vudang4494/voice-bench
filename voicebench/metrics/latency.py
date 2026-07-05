"""Tổng hợp latency. Report median + p90 (KHÔNG dùng mean — đuôi lệch phải).
Giả định caller đã loại warmup runs trước khi đưa vào."""
from __future__ import annotations

from typing import Sequence
import numpy as np


def summarize_latency(values_s: Sequence[float]) -> dict:
    a = np.asarray([v for v in values_s if v == v], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "median_s": float(np.median(a)),
        "p90_s": float(np.percentile(a, 90)),
        "p99_s": float(np.percentile(a, 99)),
        "mean_s": float(a.mean()),
        "min_s": float(a.min()),
        "max_s": float(a.max()),
    }

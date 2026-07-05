"""Timer chính xác cho GPU. Gotcha lớn nhất của đo latency ML:

CUDA/MPS chạy bất đồng bộ -> nếu không synchronize trước khi đọc perf_counter,
số đo là thời gian *enqueue* chứ không phải *execute* -> sai hoàn toàn.
Context manager này tự sync device (CUDA hoặc Apple MPS) nếu có torch + GPU.
"""
from __future__ import annotations

import time
import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


def _device_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif (getattr(torch.backends, "mps", None) is not None
              and torch.backends.mps.is_available()
              # torch cũ (<2.0) có backends.mps nhưng CHƯA có torch.mps module
              and hasattr(getattr(torch, "mps", None), "synchronize")):
            # Mac dev box: MPS cũng async — không sync thì timing lạc quan giả.
            torch.mps.synchronize()
    except ImportError:
        pass  # không có torch -> CPU-only, bỏ qua


@contextmanager
def measure() -> Iterator["list[float]"]:
    """Đo wall time (giây) của block, đã device-sync (CUDA/MPS) 2 đầu.

    Usage:
        with measure() as t:
            model(x)
        elapsed = t[0]
    """
    holder: list[float] = [float("nan")]
    _device_sync()
    start = time.perf_counter()
    try:
        yield holder
    finally:
        _device_sync()
        holder[0] = time.perf_counter() - start

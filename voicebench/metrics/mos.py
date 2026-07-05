"""Naturalness (MOS) không cần human — dùng predictor UTMOS (nếu cài).
Đây là OPTIONAL và chỉ tương quan thô với MOS người thật; đừng coi là tuyệt đối.

Trả None nếu không cài được -> harness vẫn chạy, chỉ thiếu cột MOS.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_utmos():
    try:
        import torch
        # UTMOS strong learner qua torch.hub (sarulab-speech/UTMOS22).
        model = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
        return model
    except Exception as e:  # noqa: BLE001 — nhiều loại lỗi (mạng, thiếu lib)
        logger.warning("Không nạp được UTMOS (%s) -> MOS sẽ None", e)
        return None


def predict_mos(wav: np.ndarray, sr: int) -> Optional[float]:
    model = _load_utmos()
    if model is None:
        return None
    import torch
    t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return float(model(t, sr).item())

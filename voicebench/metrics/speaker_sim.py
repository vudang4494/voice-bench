"""Voice-cloning fidelity = cosine similarity của speaker embedding
(ref voice vs cloned output). Dùng ECAPA-TDNN (SpeechBrain).

CẢNH BÁO diễn giải: cosine 0.7 tốt hay tệ KHÔNG biết nếu không có baseline.
Bắt buộc chạy calibrate_baselines(): same-speaker (2 clip thật cùng người)
và different-speaker -> mới đặt được ngưỡng. Số trần trụi vô nghĩa.

Lazy import: chỉ nạp speechbrain/torch khi thực sự gọi -> harness core import
được mà không cần GPU stack.
"""
from __future__ import annotations

import logging
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_encoder(device: str = "cuda"):
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as e:
        raise ImportError(
            "Cần 'speechbrain' + 'torch' cho speaker_sim. "
            "pip install speechbrain torch"
        ) from e
    dev = device if _torch_cuda_ok() else "cpu"
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": dev},
    )


def _torch_cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _embed(wav: np.ndarray, sr: int, device: str) -> np.ndarray:
    import torch
    enc = _load_encoder(device)
    if sr != 16000:
        import torchaudio.functional as AF
        t = torch.tensor(wav, dtype=torch.float32)
        t = AF.resample(t, sr, 16000)
    else:
        t = torch.tensor(wav, dtype=torch.float32)
    with torch.no_grad():
        emb = enc.encode_batch(t.unsqueeze(0)).squeeze().cpu().numpy()
    return emb


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def speaker_similarity(ref_wav: np.ndarray, ref_sr: int,
                       out_wav: np.ndarray, out_sr: int,
                       device: str = "cuda") -> float:
    """Cosine similarity giữa 2 waveform. [0,1] cao = giống giọng hơn."""
    ea = _embed(ref_wav, ref_sr, device)
    eb = _embed(out_wav, out_sr, device)
    return cosine(ea, eb)

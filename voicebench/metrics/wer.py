"""WER / CER qua jiwer, có chuẩn hoá VN. Corpus-level aggregation đúng
(gộp S/D/I rồi chia tổng đơn vị ref — KHÔNG lấy mean từng câu: câu ngắn over-weight).
Áp dụng cho CẢ WER (từ) lẫn CER (ký tự): corpus_wer / corpus_cer."""
from __future__ import annotations

from typing import Sequence
import jiwer

from .text_norm import normalize_vi


def compute_wer(ref: str, hyp: str, keep_tone: bool = True) -> float:
    r, h = normalize_vi(ref, keep_tone), normalize_vi(hyp, keep_tone)
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.wer(r, h)


def compute_cer(ref: str, hyp: str, keep_tone: bool = True) -> float:
    r, h = normalize_vi(ref, keep_tone), normalize_vi(hyp, keep_tone)
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.cer(r, h)


def _corpus(refs: Sequence[str], hyps: Sequence[str], keep_tone: bool,
            process, unit_key: str, rate_key: str) -> dict:
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) != hyps ({len(hyps)})")
    r = [normalize_vi(x, keep_tone) for x in refs]
    h = [normalize_vi(x, keep_tone) for x in hyps]
    out = process(r, h)
    n = out.substitutions + out.deletions + out.hits
    if n == 0 and out.insertions > 0:
        # Toàn bộ ref rỗng sau normalize nhưng hyp có nội dung: trả 0.0 sẽ chấm
        # điểm tuyệt đối cho hallucination -> data hỏng, phải fail to.
        raise ValueError(
            f"Tổng {unit_key} reference = 0 sau normalize nhưng hypothesis có "
            f"{out.insertions} insertion — manifest/ref hỏng, không tính {rate_key} được")
    rate = (out.substitutions + out.deletions + out.insertions) / n if n else 0.0
    return {
        rate_key: rate,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
        unit_key: n,
    }


def corpus_wer(refs: Sequence[str], hyps: Sequence[str], keep_tone: bool = True) -> dict:
    """WER toàn corpus = (S+D+I)/N_words_ref. Trả kèm breakdown lỗi.
    Raise ValueError nếu ref rỗng toàn bộ mà hyp có từ (hallucination không được = 0.0)."""
    return _corpus(refs, hyps, keep_tone, jiwer.process_words, "ref_words", "wer")


def corpus_cer(refs: Sequence[str], hyps: Sequence[str], keep_tone: bool = True) -> dict:
    """CER toàn corpus, pooled S/D/I trên tổng ký tự ref — cùng phương pháp corpus_wer."""
    return _corpus(refs, hyps, keep_tone, jiwer.process_characters, "ref_chars", "cer")

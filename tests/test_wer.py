import math
from voicebench.metrics.wer import compute_wer, compute_cer, corpus_wer


def test_wer_one_insertion():
    # 1 insertion / 3 từ ref = 0.3333
    assert math.isclose(compute_wer("xin chao ban", "xin chao ban nhe"),
                        1/3, rel_tol=1e-6)


def test_wer_perfect():
    assert compute_wer("xin chào bạn", "Xin chào bạn!") == 0.0  # normalize khớp


def test_cer_tone_counts_when_kept():
    # "chào" vs "chao": 1 ký tự khác / 8 = 0.125
    assert math.isclose(compute_cer("xin chào", "xin chao"), 0.125, rel_tol=1e-6)


def test_cer_tone_ignored_when_stripped():
    assert compute_cer("xin chào", "xin chao", keep_tone=False) == 0.0


def test_corpus_aggregation():
    refs = ["xin chao ban", "toi hoc bai"]
    hyps = ["xin chao ban", "toi hoc bai roi"]  # 1 insertion tổng
    out = corpus_wer(refs, hyps)
    assert out["ref_words"] == 6
    assert out["insertions"] == 1
    assert math.isclose(out["wer"], 1/6, rel_tol=1e-6)


def test_corpus_length_mismatch_raises():
    try:
        corpus_wer(["a"], ["a", "b"])
        assert False, "phải raise"
    except ValueError:
        pass


# --- T1 fix: corpus CER pooled + guard ref rỗng ---

def test_corpus_cer_pooled_not_mean():
    from voicebench.metrics.wer import corpus_cer
    # pooled: (1+1) lỗi / (4+1) ký tự ref = 0.4; mean-per-sentence sẽ là 0.625
    out = corpus_cer(["aaaa", "a"], ["aaab", "b"])
    assert math.isclose(out["cer"], 2 / 5, rel_tol=1e-9)
    assert out["ref_chars"] == 5


def test_corpus_wer_all_empty_refs_with_hyp_raises():
    # ref toàn dấu câu -> normalize rỗng; hyp có từ -> phải raise, không trả 0.0
    try:
        corpus_wer(["...", "!!"], ["xin chao", "um"])
        assert False, "phải raise ValueError"
    except ValueError:
        pass


def test_corpus_wer_all_empty_both_ok():
    out = corpus_wer(["...", "!!"], ["", "?"])
    assert out["wer"] == 0.0 and out["ref_words"] == 0

from voicebench.metrics.text_norm import normalize_vi


def test_lowercase_and_punct():
    assert normalize_vi("Xin Chào, bạn!") == "xin chào bạn"


def test_collapse_whitespace():
    assert normalize_vi("  a   b\tc\n") == "a b c"


def test_nfc_normalization():
    # decomposed "cà" (c + a + U+0300 combining grave) phải == precomposed "cà"
    decomposed = "ca\u0300"
    assert normalize_vi(decomposed) == normalize_vi("cà") == "cà"


def test_keep_tone_default():
    # giữ dấu thanh: "chào" != "chao"
    assert normalize_vi("chào") == "chào"


def test_strip_tone_optional():
    assert normalize_vi("chào bạn", keep_tone=False) == "chao ban"


def test_none_safe():
    assert normalize_vi(None) == ""


# --- T1 fix: strip_tone chỉ bỏ 5 dấu thanh, giữ dấu phụ nguyên âm + đ ---

def test_strip_tone_preserves_vowel_quality_diacritics():
    # ế -> ê (bỏ sắc, giữ mũ), ậ -> â, ẹ -> e; đ giữ nguyên
    assert normalize_vi("tiếng Việt đẹp", keep_tone=False) == "tiêng viêt đep"


def test_strip_tone_does_not_merge_vowel_quality():
    # 'cần' -> 'cân' phải KHÁC 'căn' -> 'căn' và KHÁC 'can'
    assert normalize_vi("cần", keep_tone=False) == "cân"
    assert normalize_vi("căn", keep_tone=False) == "căn"
    assert normalize_vi("cắn", keep_tone=False) == "căn"


def test_strip_tone_preserves_horn_and_breve():
    assert normalize_vi("thư", keep_tone=False) == "thư"
    assert normalize_vi("thử", keep_tone=False) == "thư"
    assert normalize_vi("ăn", keep_tone=False) == "ăn"


def test_strip_tone_output_is_nfc():
    import unicodedata
    out = normalize_vi("cần thơ", keep_tone=False)
    assert out == unicodedata.normalize("NFC", out)


def test_orphan_combining_mark_not_word_boundary():
    # 'İ'.lower() = i + U+0307 (không precompose được): mark bị xoá,
    # KHÔNG thành space tách từ ('i stanbul' là sai)
    assert normalize_vi("İstanbul") == "istanbul"


# --- fold_spelling_variants: biến thể chính tả hợp lệ, KHÔNG phải lỗi ASR ---

def test_fold_tone_placement_oa_oe_uy():
    from voicebench.metrics.text_norm import fold_spelling_variants as fold
    # kiểu cũ (dấu trên nguyên âm trước) == kiểu mới (dấu trên nguyên âm sau)
    assert fold("hóa") == fold("hoá")
    assert fold("khỏe") == fold("khoẻ")
    assert fold("thúy") == fold("thuý")
    assert fold("ủy ban") == fold("uỷ ban")


def test_fold_i_y_after_onset():
    from voicebench.metrics.text_norm import fold_spelling_variants as fold
    assert fold("kì") == fold("kỳ")
    assert fold("lí") == fold("lý")
    assert fold("mĩ") == fold("mỹ")
    assert fold("quí") == fold("quý")
    assert fold("tị nạn") == fold("tỵ nạn")


def test_fold_khong_dong_cham_tu_khac_nhau():
    from voicebench.metrics.text_norm import fold_spelling_variants as fold
    # ay/ai là TỪ khác nhau, không phải biến thể
    assert fold("hay") != fold("hai")
    # tone trên 'o' đứng một mình (nói) không nằm trong cụm oa/oe -> giữ nguyên
    assert fold("nói") == "nói"
    # thuở/thủa là biến thể từ vựng -> KHÔNG fold
    assert fold("thuở") != fold("thủa")
    # 'ý' không có phụ âm đầu -> giữ nguyên; 'ngày' y sau nguyên âm -> giữ nguyên
    assert fold("ý") == "ý"
    assert fold("ngày") == "ngày"


def test_fold_variants_trong_corpus_wer():
    from voicebench.metrics.wer import corpus_wer
    refs, hyps = ["quy hoạch kỳ lạ"], ["quy hoạch kì lạ"]
    assert corpus_wer(refs, hyps)["wer"] > 0          # số chính: vẫn tính là lỗi
    assert corpus_wer(refs, hyps, fold_variants=True)["wer"] == 0.0  # số phụ

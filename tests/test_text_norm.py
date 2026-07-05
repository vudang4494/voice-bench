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

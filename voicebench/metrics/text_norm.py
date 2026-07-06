"""Chuẩn hoá text tiếng Việt trước khi tính WER/CER.

Quy tắc (chuẩn ASR eval + đặc thù VN):
- NFC unicode: gộp dấu thanh về precomposed (ạ, ế...) — bắt buộc.
- lowercase.
- bỏ dấu câu, GIỮ chữ + dấu thanh (\\w Python3 unicode-aware, giữ á à ạ đ ...).
- collapse whitespace.

CHÚ Ý: giữ dấu thanh mặc định. Bỏ dấu thanh sẽ che lỗi tone của model -> sai mục đích đo.

keep_tone=False chỉ bỏ 5 DẤU THANH (sắc/huyền/hỏi/ngã/nặng). Dấu PHỤ nguyên âm
(â ă ê ô ơ ư — circumflex/breve/horn) và đ KHÔNG phải dấu thanh, phải giữ:
'cần' -> 'cân' (khác 'căn'/'can'), 'thư' -> 'thư', 'đẹp' -> 'đep'.
Strip toàn bộ Mn mark là SAI: gộp 'cân'/'căn'/'can' làm một, đo tone error thành vô nghĩa.

Giới hạn đã biết (chấp nhận): strip_tone lọc theo codepoint sau NFD nên dấu
TRÙNG codepoint trên chữ ngoại lai cũng bị bỏ (ñ->n vì tilde = U+0303, ź->z vì
acute = U+0301). Đối xứng ref/hyp, chỉ ảnh hưởng từ mượn ở mode keep_tone=False
— không đáng đổi lấy việc phải phân loại base letter. Xem ROADMAP T9.

Combining mark "mồ côi" (không precompose được, vd 'İ'.lower() = i + U+0307):
bị XOÁ chứ không thay bằng space — mark không bao giờ là ranh giới từ.
"""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)

# 5 dấu thanh VN ở dạng combining (sau NFD): sắc, huyền, hỏi, ngã, nặng.
_TONE_MARKS = frozenset({
    "\u0301",  # sắc
    "\u0300",  # huyền
    "\u0309",  # hỏi
    "\u0303",  # ngã
    "\u0323",  # nặng
})


def strip_tone(text: str) -> str:
    """Bỏ ĐÚNG 5 dấu thanh, giữ dấu phụ nguyên âm (â ă ê ô ơ ư) và đ; trả về NFC."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if c not in _TONE_MARKS))


def _punct_to_space(m: "re.Match[str]") -> str:
    # Mn mồ côi (sau lower(), vd i+U+0307) xoá hẳn; punct thật -> space.
    return "" if unicodedata.category(m.group()) == "Mn" else " "


# --- Biến thể chính tả hợp lệ (T9): hóa/hoá, kì/kỳ — KHÔNG phải lỗi ASR. ---
# Fold là OPT-IN (số WER phụ). Chỉ 2 lớp biến thể được cộng đồng chấp nhận:
# 1) Vị trí dấu thanh kiểu cũ/mới trên cụm oa/oe/uy (hóa/hoá, khỏe/khoẻ, thúy/thuý)
#    -> canonical: dấu đặt trên nguyên âm SAU (kiểu mới). Làm trên NFD.
#    Cụm ua/oi/ay... KHÔNG fold: "hay"/"hai", "nói" là từ khác nhau, không phải biến thể.
#    "thuở/thủa" là biến thể TỪ VỰNG, không fold (horn chắn giữa o và dấu thanh nên regex né sẵn).
# 2) i/y là nguyên âm duy nhất sau phụ âm đầu h/k/l/m/s/t/qu (kì/kỳ, lí/lý, quí/quý)
#    -> canonical: i. KHÔNG áp cho y sau nguyên âm (ay/uy nguyên cụm) hay "ý" đứng đầu từ.
_TONE_OA_OE = re.compile(r"o([̣̀́̃̉])([ae])")
_TONE_UY = re.compile(r"u([̣̀́̃̉])(y)")
_Y_WORD = re.compile(r"\b(qu|[hklmst])([yỳýỷỹỵ])\b")
_Y2I = {"y": "i", "ỳ": "ì", "ý": "í", "ỷ": "ỉ", "ỹ": "ĩ", "ỵ": "ị"}


def fold_spelling_variants(text: str) -> str:
    """Gộp biến thể chính tả hợp lệ về 1 dạng canonical để chúng không bị tính
    là lỗi. Input phải đã qua normalize_vi (lowercase, NFC). Trả về NFC."""
    d = unicodedata.normalize("NFD", text)
    d = _TONE_OA_OE.sub(r"o\2\1", d)
    d = _TONE_UY.sub(r"u\2\1", d)
    text = unicodedata.normalize("NFC", d)
    return _Y_WORD.sub(lambda m: m.group(1) + _Y2I[m.group(2)], text)


def normalize_vi(text: str, keep_tone: bool = True) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _PUNCT.sub(_punct_to_space, text)
    if not keep_tone:
        text = strip_tone(text)
    text = _WS.sub(" ", text).strip()
    return text

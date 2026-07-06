"""Test verifier dataset ASR train (scripts/verify_asr_dataset.py) — không cần model."""
import json

import numpy as np
import pytest
import soundfile as sf
import yaml

from scripts.verify_asr_dataset import ROOT, Verifier, load_manifest, norm_text


@pytest.fixture(scope="module")
def rules():
    with open(f"{ROOT}/configs/data_rules.asr.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _wav(path, dur_s=2.0, sr=16000, amp=0.3):
    t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False)
    sf.write(str(path), (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr,
             subtype="PCM_16")
    return str(path)


def test_norm_text_chuan():
    # NFC + lowercase + bỏ punct + gộp whitespace lạ (NBSP)
    assert norm_text("Tiếng Việt, RẤT  hay!") == "tiếng việt rất hay"
    # NFD input phải về NFC trước khi so
    nfd = "tiếng"  # 'tiếng' dạng NFD
    assert norm_text(nfd) == norm_text("tiếng")


def test_text_gates_bat_chu_hoa_va_so(rules, tmp_path):
    v = Verifier(rules)
    rows = [{"id": "a", "text": "Xin Chào ANH"},   # chữ hoa
            {"id": "b", "text": "có 5 quả cam"},   # chữ số
            {"id": "c", "text": "câu sạch bình thường"}]
    v.check_text(rows)
    got = {rid: ok for rid, ok, _ in v.results}
    assert got["TXT-uppercase"] is False   # 33% > 0%
    assert got["TXT-digit"] is False       # 33% > 0.5%
    assert got["TXT-nfc"] is True


def test_text_gates_pass_khi_sach(rules):
    v = Verifier(rules)
    rows = [{"id": str(i), "text": f"câu tiếng việt sạch số {'x' * (i + 1)}"}
            for i in range(10)]
    v.check_text(rows)
    assert all(ok for _, ok, _ in v.results)


def test_audio_gates(rules, tmp_path):
    v = Verifier(rules)
    ok_wav = _wav(tmp_path / "ok.wav")
    quiet = _wav(tmp_path / "quiet.wav", amp=0.001)          # RMS < 0.005
    rows = [{"id": "ok", "audio": ok_wav, "text": "một câu vừa đủ dài ổn"},
            {"id": "quiet", "audio": quiet, "text": "một câu vừa đủ dài ổn"}]
    v.check_audio(rows)
    got = {rid: ok for rid, ok, _ in v.results}
    assert got["AUD-samplerate"] is True
    assert got["AUD-rms"] is False          # 50% quiet > 1%


def test_integrity_bat_caption_spam_va_audio_dup(rules, tmp_path):
    v = Verifier(rules)
    w1 = _wav(tmp_path / "w1.wav")
    w2 = _wav(tmp_path / "w2.wav", dur_s=1.5)
    spam = [{"id": f"s{i}", "audio": w1, "text": "đăng ký kênh nhé mọi người"}
            for i in range(6)]                                # 1 text lặp 6 lần + audio dup
    rows = spam + [{"id": "z", "audio": w2, "text": "một câu khác hẳn nội dung"}]
    v.check_integrity(rows)
    got = {rid: ok for rid, ok, _ in v.results}
    assert got["INT-rep-text"] is False
    assert got["INT-dup-audio"] is False
    assert got["INT-eval-contam"] is True


def test_eval_contam_bat_trung_ref_eval(rules, tmp_path):
    # Lấy 1 ref thật từ eval set của product -> phải bị bắt
    ref = json.loads(open(f"{ROOT}/data/eval/eval_3min.manifest.jsonl",
                          encoding="utf-8").readline())["text"]
    w = _wav(tmp_path / "w.wav")
    v = Verifier(rules)
    v.check_integrity([{"id": "x", "audio": w, "text": ref}])
    got = {rid: ok for rid, ok, _ in v.results}
    assert got["INT-eval-contam"] is False


def test_load_manifest_ho_tro_text_raw(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps([{"id": "a", "audio": "x.wav", "text_raw": "abc"}]),
                 encoding="utf-8")
    rows = load_manifest(str(p))
    assert rows[0]["text"] == "abc"

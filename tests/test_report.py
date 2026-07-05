"""T1 fix: corpus-pooled CER trong aggregate + CLI --keep-tone/--no-keep-tone
+ default keep_tone đọc từ run_meta.json."""
import json
import sys

from voicebench import report


def _row(i, ref, asr, rt):
    lat = {"total_s": 0.1, "media_dur_s": 1.0, "model_load_s": 0.0,
           "ttfa_s": None, "first_token_s": None, "rtf": 0.1}
    return {"sample_id": str(i), "ref_text": ref, "asr_text": asr,
            "roundtrip_text": rt, "asr_latency": lat, "tts_latency": dict(lat),
            "speaker_sim": None, "mos": None, "extra": {}}


def test_aggregate_keep_tone_changes_wer():
    rows = [_row(0, "cần", "cân", "cần")]
    assert report.aggregate(rows, keep_tone=True)["asr_wer"] == 1.0
    # keep_tone=False: 'cần' -> 'cân' == hyp -> hết lỗi
    assert report.aggregate(rows, keep_tone=False)["asr_wer"] == 0.0


def test_aggregate_cer_is_corpus_pooled():
    # câu dài đúng hết + câu 1-ký-tự sai hết: mean-per-sentence sẽ ~0.5,
    # pooled phải là 1/(1+10)
    rows = [_row(0, "a", "b", "a"), _row(1, "xin chao b", "xin chao b", "xin chao b")]
    agg = report.aggregate(rows, keep_tone=True)
    assert abs(agg["asr_cer"] - 1 / 11) < 1e-9


def _write_results(tmp_path, rows):
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                 encoding="utf-8")
    return p


def test_cli_no_keep_tone_flag(tmp_path, monkeypatch, capsys):
    p = _write_results(tmp_path, [_row(0, "cần", "cân", "cần")])
    monkeypatch.setattr(sys, "argv", ["report", "-r", str(p), "--no-keep-tone"])
    report.main()
    capsys.readouterr()
    agg = json.loads((tmp_path / "results.agg.json").read_text(encoding="utf-8"))
    assert agg["keep_tone"] is False
    assert agg["asr_wer"] == 0.0


def test_cli_keep_tone_flag(tmp_path, monkeypatch, capsys):
    p = _write_results(tmp_path, [_row(0, "cần", "cân", "cần")])
    monkeypatch.setattr(sys, "argv", ["report", "-r", str(p), "--keep-tone"])
    report.main()
    capsys.readouterr()
    agg = json.loads((tmp_path / "results.agg.json").read_text(encoding="utf-8"))
    assert agg["keep_tone"] is True
    assert agg["asr_wer"] == 1.0


def test_cli_default_keep_tone_from_run_meta(tmp_path, monkeypatch, capsys):
    p = _write_results(tmp_path, [_row(0, "cần", "cân", "cần")])
    (tmp_path / "run_meta.json").write_text(
        json.dumps({"config": {"keep_tone": False}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["report", "-r", str(p)])  # không flag
    report.main()
    capsys.readouterr()
    agg = json.loads((tmp_path / "results.agg.json").read_text(encoding="utf-8"))
    assert agg["keep_tone"] is False


def test_cli_default_keep_tone_no_run_meta(tmp_path, monkeypatch, capsys):
    p = _write_results(tmp_path, [_row(0, "cần", "cân", "cần")])
    monkeypatch.setattr(sys, "argv", ["report", "-r", str(p)])
    report.main()
    capsys.readouterr()
    agg = json.loads((tmp_path / "results.agg.json").read_text(encoding="utf-8"))
    assert agg["keep_tone"] is True  # fallback mặc định giữ dấu thanh

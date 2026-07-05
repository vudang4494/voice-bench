"""measure() phải device-sync 2 đầu và trả wall time của block."""
import time

from voicebench import timing


def test_measure_syncs_both_ends(monkeypatch):
    calls = []
    monkeypatch.setattr(timing, "_device_sync", lambda: calls.append(1))
    with timing.measure() as t:
        pass
    assert len(calls) == 2  # sync trước start VÀ trước stop
    assert t[0] >= 0.0


def test_measure_captures_elapsed():
    with timing.measure() as t:
        time.sleep(0.02)
    assert t[0] >= 0.02


def test_measure_records_even_on_exception():
    try:
        with timing.measure() as t:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert t[0] == t[0] and t[0] >= 0.0  # không NaN


def test_device_sync_survives_old_torch_without_mps_module(monkeypatch):
    # torch 1.12/1.13 Apple Silicon: có backends.mps.is_available()==True
    # nhưng CHƯA có torch.mps -> không được crash AttributeError
    import sys
    import types

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with timing.measure() as t:
        pass
    assert t[0] >= 0.0

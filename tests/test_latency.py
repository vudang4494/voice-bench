import math
from voicebench.metrics.latency import summarize_latency
from voicebench.interfaces import LatencyBreakdown


def test_summarize_basic():
    s = summarize_latency([1, 2, 3, 4, 5])
    assert s["n"] == 5
    assert s["median_s"] == 3.0
    assert s["max_s"] == 5.0
    assert s["min_s"] == 1.0


def test_summarize_drops_nan():
    s = summarize_latency([1.0, float("nan"), 3.0])
    assert s["n"] == 2


def test_summarize_empty():
    assert summarize_latency([])["n"] == 0


def test_rtf_property():
    lb = LatencyBreakdown(total_s=2.0, media_dur_s=4.0)
    assert lb.rtf == 0.5


def test_rtf_nan_on_zero_dur():
    lb = LatencyBreakdown(total_s=2.0, media_dur_s=0.0)
    assert math.isnan(lb.rtf)


def test_to_dict_has_rtf_and_streaming_fields():
    d = LatencyBreakdown(total_s=1.0, media_dur_s=2.0).to_dict()
    assert d["rtf"] == 0.5
    assert d["ttfa_s"] is None and d["first_token_s"] is None  # forward-compat

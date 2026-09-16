"""
Unit tests for PhoneCamCapture — the browser-pushed frame buffer used by
the phone camera "Shared Perception" feature.
"""

import os
import sys
import time
import numpy as np
import pytest

_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from dashboard.stream_server import PhoneCamCapture


def _make_frame(value=128):
    return np.full((360, 640, 3), value, dtype=np.uint8)


def test_starts_with_no_frame():
    cap = PhoneCamCapture("device-1")
    assert cap.start() is True
    assert cap.is_opened is True
    assert cap.is_video_file is False
    ret, frame = cap.read_latest()
    assert ret is False
    assert frame is None


def test_push_then_read_latest():
    cap = PhoneCamCapture("device-2")
    cap.start()
    frame = _make_frame(200)
    cap.push_frame(frame)

    ret, out = cap.read_latest()
    assert ret is True
    assert out is not None
    assert np.array_equal(out, frame)
    assert cap.frame_count == 1


def test_read_latest_returns_a_copy_not_the_original():
    cap = PhoneCamCapture("device-3")
    cap.start()
    frame = _make_frame(50)
    cap.push_frame(frame)

    _, out = cap.read_latest()
    out[0, 0] = 255
    _, out2 = cap.read_latest()
    assert out2[0, 0, 0] == 50  # internal buffer untouched by caller mutation


def test_stale_frame_is_treated_as_no_frame():
    cap = PhoneCamCapture("device-4")
    cap.start()
    cap.push_frame(_make_frame())
    cap.STALE_FRAME_TIMEOUT_SECONDS = 0.05
    time.sleep(0.1)

    ret, out = cap.read_latest()
    assert ret is False
    assert out is None


def test_stop_clears_state():
    cap = PhoneCamCapture("device-5")
    cap.start()
    cap.push_frame(_make_frame())
    cap.stop()

    assert cap.is_opened is False
    ret, out = cap.read_latest()
    assert ret is False


if __name__ == "__main__":
    pytest.main(["-v", __file__])

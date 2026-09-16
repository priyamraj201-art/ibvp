"""
Verifies CameraPipelineWorker picks PhoneCamCapture for phonecam:// URLs
and ZeroLagCapture for everything else, without starting real inference.
"""

import os
import sys
import time
import numpy as np
import pytest

_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from dashboard.stream_server import (
    CameraPipelineWorker,
    CameraStreamState,
    PhoneCamCapture,
    ZeroLagCapture,
)
from dashboard.phonecam import PHONE_DEVICE_REGISTRY


def test_phonecam_url_selects_phone_cam_capture():
    device_id = "worker-test-device"
    # Pre-seed a frame so the worker's loop has something to read immediately.
    cap = PHONE_DEVICE_REGISTRY.get_or_create_capture(device_id)
    cap.start()
    cap.push_frame(np.full((360, 640, 3), 100, dtype=np.uint8))

    cam_info = {"id": 999, "name": "Phone Test Cam", "location": "Test", "url": f"phonecam://{device_id}"}
    state = CameraStreamState(cam_id=999, name="Phone Test Cam")
    worker = CameraPipelineWorker(cam_info, {"enable_frs": False, "enable_anpr": False}, state)

    resolved_cap = worker._resolve_capture()
    assert isinstance(resolved_cap, PhoneCamCapture)
    assert resolved_cap is cap


def test_regular_url_still_selects_zero_lag_capture():
    cam_info = {"id": 998, "name": "File Test Cam", "location": "Test", "url": "videos/palace.mp4"}
    state = CameraStreamState(cam_id=998, name="File Test Cam")
    worker = CameraPipelineWorker(cam_info, {"enable_frs": False, "enable_anpr": False}, state)

    resolved_cap = worker._resolve_capture()
    assert isinstance(resolved_cap, ZeroLagCapture)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

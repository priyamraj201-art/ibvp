"""
Integration tests for the phone camera WebSocket frame-ingest endpoint.
"""

import os
import sys
import cv2
import numpy as np
import pytest

_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from dashboard.server import create_app
from dashboard.phonecam import PHONE_DEVICE_REGISTRY
from dashboard.stream_server import MultiCameraManager
from dashboard.camera_manager import CameraManager
from fastapi.testclient import TestClient


def _client():
    frs_db = os.path.join(_bytetrack_root, "frs_faces.db")
    anpr_db = os.path.join(_bytetrack_root, "anpr_watchlist.db")
    return TestClient(create_app(frs_db, anpr_db))


def _jpeg_bytes(value=77):
    frame = np.full((240, 320, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def test_websocket_rejects_invalid_token():
    client = _client()
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/phonecam/bad-token/device-x?name=Test"):
            pass


def test_websocket_accepts_valid_token_and_feeds_capture(tmp_path, monkeypatch):
    # Isolate the real, shared CAMERA_REGISTRY / MULTI_CAMERA_MANAGER singletons
    # before connecting to the real websocket route, which internally calls
    # PHONE_DEVICE_REGISTRY.register_device(...) -> CAMERA_REGISTRY.upsert_camera(...)
    # and MULTI_CAMERA_MANAGER.start_camera(...). Without this, the test would
    # write a real camera entry into the production cameras.json and spawn a
    # real background CameraPipelineWorker thread against a fake phonecam:// URL.
    cfg_file = str(tmp_path / "test_cameras.json")
    isolated_camera_registry = CameraManager(config_path=cfg_file)
    isolated_multi_camera_manager = MultiCameraManager()

    import dashboard.phonecam
    import dashboard.stream_server
    monkeypatch.setattr(dashboard.phonecam, "CAMERA_REGISTRY", isolated_camera_registry)
    monkeypatch.setattr(dashboard.phonecam, "MULTI_CAMERA_MANAGER", isolated_multi_camera_manager)
    monkeypatch.setattr(dashboard.stream_server, "CAMERA_REGISTRY", isolated_camera_registry)

    client = _client()
    token = PHONE_DEVICE_REGISTRY.create_pair_token()
    device_id = "ws-test-device"

    with client.websocket_connect(f"/ws/phonecam/{token}/{device_id}?name=Gate%20Test") as ws:
        ws.send_bytes(_jpeg_bytes())
        ws.close()

    capture = PHONE_DEVICE_REGISTRY.get_or_create_capture(device_id)
    ret, frame = capture.read_latest()
    assert ret is True
    assert frame.shape[2] == 3


if __name__ == "__main__":
    pytest.main(["-v", __file__])

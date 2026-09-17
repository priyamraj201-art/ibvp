"""
Integration tests for the phone camera WebSocket frame-ingest endpoint.
"""

import os
import sys
import threading
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


def _isolate_camera_singletons(tmp_path, monkeypatch):
    """Isolate the real, shared CAMERA_REGISTRY / MULTI_CAMERA_MANAGER singletons
    before connecting to the real websocket route, which internally calls
    PHONE_DEVICE_REGISTRY.register_device(...) -> CAMERA_REGISTRY.upsert_camera(...)
    and MULTI_CAMERA_MANAGER.start_camera(...). Without this, the test would
    write a real camera entry into the production cameras.json and spawn a
    real background CameraPipelineWorker thread against a fake phonecam:// URL.

    Returns the isolated MultiCameraManager instance so the caller can stop
    (and join) any worker thread it spawns once the test is done with it —
    otherwise a real CameraPipelineWorker daemon thread (which loads the real
    GPU OSNet ReID model via the unpatched SHARED_DETECTOR singleton) keeps
    running, leaked, for the rest of the pytest session.
    """
    cfg_file = str(tmp_path / "test_cameras.json")
    isolated_camera_registry = CameraManager(config_path=cfg_file)
    isolated_multi_camera_manager = MultiCameraManager()

    import dashboard.phonecam
    import dashboard.stream_server
    monkeypatch.setattr(dashboard.phonecam, "CAMERA_REGISTRY", isolated_camera_registry)
    monkeypatch.setattr(dashboard.phonecam, "MULTI_CAMERA_MANAGER", isolated_multi_camera_manager)
    monkeypatch.setattr(dashboard.stream_server, "CAMERA_REGISTRY", isolated_camera_registry)

    return isolated_camera_registry, isolated_multi_camera_manager


def _stop_and_join_workers(multi_camera_manager, timeout=30):
    """Stop every worker the isolated MultiCameraManager started and actually
    join() each thread before returning, so the test doesn't leave a real
    CameraPipelineWorker daemon thread (GPU model load included) running past
    this test. MultiCameraManager.stop_camera()/.stop_all() only signal the
    worker to stop; they don't join it, so we grab the thread objects first
    and join them ourselves.
    """
    with multi_camera_manager._lock:
        workers = list(multi_camera_manager.workers.values())
    multi_camera_manager.stop_all()
    for worker in workers:
        worker.join(timeout=timeout)


@pytest.fixture(scope="module", autouse=True)
def _camworker_threads_before_module():
    """Snapshot of CamWorker-* thread idents already alive (e.g. leaked by an
    earlier test module such as test_phonecam_registry.py's register_device
    tests, when this file runs as part of the full phonecam suite) before any
    test in *this* module runs. Used by
    test_no_new_leaked_camworker_threads_after_module to check only for
    leaks newly introduced by this module."""
    return {t.ident for t in threading.enumerate() if t.name.startswith("CamWorker-")}


def test_websocket_rejects_invalid_token():
    # No registration happens on this path (token check fails before
    # PHONE_DEVICE_REGISTRY.register_device is ever called), so no isolation
    # or worker-thread cleanup is needed here.
    client = _client()
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/phonecam/bad-token/device-x?name=Test"):
            pass


def test_websocket_accepts_valid_token_and_feeds_capture(tmp_path, monkeypatch):
    _isolated_camera_registry, isolated_multi_camera_manager = _isolate_camera_singletons(
        tmp_path, monkeypatch
    )

    client = _client()
    token = PHONE_DEVICE_REGISTRY.create_pair_token()
    device_id = "ws-test-device"

    try:
        with client.websocket_connect(f"/ws/phonecam/{token}/{device_id}?name=Gate%20Test") as ws:
            ws.send_bytes(_jpeg_bytes())
            # Grab the capture reference while still connected: on disconnect
            # the WebSocket handler unregisters the device (removing it from
            # the dashboard), so a post-close get_or_create_capture() would
            # hand back a brand-new, empty capture instead of this one.
            capture = PHONE_DEVICE_REGISTRY.get_or_create_capture(device_id)
            ws.close()

        ret, frame = capture.read_latest()
        assert ret is True
        assert frame.shape[2] == 3
    finally:
        # Stop and join the real CameraPipelineWorker thread that
        # register_device() spawned via the isolated MULTI_CAMERA_MANAGER,
        # so no worker thread is leaked past this test.
        _stop_and_join_workers(isolated_multi_camera_manager)


def test_websocket_drops_corrupt_frame_without_crashing(tmp_path, monkeypatch):
    """A corrupt/undecodable frame (cv2.imdecode returns None) must be
    dropped, not crash the WebSocket handler — mirroring how ZeroLagCapture
    already tolerates bad reads from network streams. Prove the handler
    survives it by sending garbage bytes, then a valid JPEG frame, and
    confirming the capture ends up holding the valid frame."""
    _isolated_camera_registry, isolated_multi_camera_manager = _isolate_camera_singletons(
        tmp_path, monkeypatch
    )

    client = _client()
    token = PHONE_DEVICE_REGISTRY.create_pair_token()
    device_id = "ws-test-device-corrupt"

    try:
        with client.websocket_connect(f"/ws/phonecam/{token}/{device_id}?name=Gate%20Test") as ws:
            ws.send_bytes(b"not a real jpeg")  # undecodable -> cv2.imdecode returns None
            ws.send_bytes(_jpeg_bytes(value=99))  # handler must still be alive to process this
            # Grab the capture reference while still connected — see comment
            # in test_websocket_accepts_valid_token_and_feeds_capture above.
            capture = PHONE_DEVICE_REGISTRY.get_or_create_capture(device_id)
            ws.close()

        ret, frame = capture.read_latest()
        assert ret is True
        assert frame.shape[2] == 3
    finally:
        _stop_and_join_workers(isolated_multi_camera_manager)


def test_no_new_leaked_camworker_threads_after_module(_camworker_threads_before_module):
    """Guard against regressions in the two registering tests above: after
    they've each run and cleaned up via _stop_and_join_workers, this module
    must not have left any *new* CameraPipelineWorker ("CamWorker-*") thread
    alive.

    Compares against `_camworker_threads_before_module`, a baseline snapshot
    taken before this module's own tests ran, rather than asserting zero
    CamWorker threads process-wide: when this file runs as part of the full
    phonecam suite, test_phonecam_registry.py's (Task 2, already-merged,
    out-of-scope-for-this-fix) register_device tests independently leak their
    own CamWorker threads via the same unpatched MULTI_CAMERA_MANAGER.start_camera
    call path. Those are a pre-existing issue logged separately for the final
    whole-branch review, not something this test should fail on. This test
    only proves *this module's* tests aren't leaking on top of that.

    Relies on running last in this file (pytest's default top-to-bottom
    collection order within a module).
    """
    current = {t.ident: t.name for t in threading.enumerate() if t.name.startswith("CamWorker-")}
    new_leaked = [
        name for ident, name in current.items()
        if ident not in _camworker_threads_before_module
    ]
    assert new_leaked == [], f"This module leaked new CameraPipelineWorker thread(s): {new_leaked}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

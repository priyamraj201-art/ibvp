"""
Unit & Integration Test Suite for Multi-Camera Surveillance Cluster.
Validates:
1. cameras.json persistent registry CRUD
2. ZeroLagCapture source resolution (DroidCam, RTSP, Webcams, Video Files)
3. MultiCameraManager isolated states & tracking instances
4. FastAPI endpoints for multi-camera cluster control & streaming
"""

import os
import sys
import json
import pytest

# Ensure ByteTrack root is on path
_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from dashboard.camera_manager import CameraManager, DEFAULT_CAMERAS
from dashboard.stream_server import (
    ZeroLagCapture,
    MultiCameraManager,
    CameraStreamState,
    create_placeholder_jpeg,
)
from dashboard.server import create_app
from fastapi.testclient import TestClient


def test_cameras_json_registry(tmp_path):
    """Test CameraManager loading, saving, upserting, deleting."""
    cfg_file = str(tmp_path / "test_cameras.json")
    mgr = CameraManager(config_path=cfg_file)

    # Initial load should create defaults
    cams = mgr.load_cameras()
    assert len(cams) == 4
    assert cams[0]["name"] == "CAM 01: Checkpost Alpha"
    assert "4747" in cams[0]["url"]

    # Upsert new camera
    new_cam = mgr.upsert_camera({
        "id": 99,
        "name": "CAM 99: Test DroidCam",
        "location": "Sector 9",
        "url": "http://192.168.1.50:4747/video",
        "enabled": True,
    })
    assert new_cam["id"] == 99
    assert mgr.get_camera(99) is not None

    # Toggle enabled
    toggled = mgr.toggle_camera(99, enabled=False)
    assert toggled["enabled"] is False

    # Delete camera
    deleted = mgr.delete_camera(99)
    assert deleted is True
    assert mgr.get_camera(99) is None


def test_zero_lag_capture_source_resolution():
    """Verify source resolution logic for webcams, RTSP, DroidCam, and video files."""
    # Test numeric index
    cap_webcam = ZeroLagCapture("0")
    assert cap_webcam.source_str == "0"

    # Test DroidCam URL
    cap_droid = ZeroLagCapture("http://192.168.137.15:4747/video")
    assert "4747" in cap_droid.source_str

    # Test RTSP URL
    cap_rtsp = ZeroLagCapture("rtsp://admin:pass@192.168.1.100:554/live")
    assert cap_rtsp.source_str.startswith("rtsp://")

    # Test Video File path
    cap_video = ZeroLagCapture("videos/palace.mp4")
    assert cap_video.source_str == "videos/palace.mp4"


def test_placeholder_jpeg_generation():
    """Verify offline / connecting placeholder generation returns valid JPEG bytes."""
    jpeg = create_placeholder_jpeg("CAM 01: Test", "OFFLINE")
    assert isinstance(jpeg, bytes)
    assert len(jpeg) > 100
    # Check JPEG magic bytes: FF D8 FF
    assert jpeg[:3] == b"\xff\xd8\xff"


def test_multi_camera_manager_states():
    """Verify MultiCameraManager isolated states per camera ID."""
    mgr = MultiCameraManager()
    state_0 = mgr.get_state(0)
    state_1 = mgr.get_state(1)

    assert state_0.stats.cam_id == 0
    assert state_1.stats.cam_id == 1
    assert state_0 is not state_1

    # Update stats for camera 0 only
    state_0.update_stats(fps=29.5, human_count=3, active_tracks=5)
    assert state_0.stats.fps == 29.5
    assert state_0.stats.human_count == 3
    assert state_1.stats.human_count == 0  # Camera 1 isolated!

    stats_all = mgr.get_all_stats()
    assert "cameras" in stats_all
    assert "cluster_metrics" in stats_all
    assert len(stats_all["cameras"]) >= 4


def test_fastapi_endpoints():
    """Verify all FastAPI routes for live streaming, cameras, and backward compatibility."""
    frs_db = os.path.join(_bytetrack_root, "frs_faces.db")
    anpr_db = os.path.join(_bytetrack_root, "anpr_watchlist.db")
    app = create_app(frs_db, anpr_db)
    client = TestClient(app)

    # 1. Test live dashboard HTML page
    res = client.get("/live")
    assert res.status_code == 200
    assert "Multi-Camera Surveillance Cluster" in res.text
    assert "2x2 Grid View" in res.text
    assert "1x1 Focus View" in res.text

    # 2. Test get cameras API
    res = client.get("/api/cameras")
    assert res.status_code == 200
    data = res.json()
    assert "cameras" in data
    assert len(data["cameras"]) >= 4

    # 3. Test get all stats API
    res = client.get("/api/cameras/stats_all")
    assert res.status_code == 200
    stats_data = res.json()
    assert "cluster_metrics" in stats_data

    # 4. Test single camera stats API
    res = client.get("/api/cameras/0/stats")
    assert res.status_code == 200
    assert res.json()["cam_id"] == 0

    # 5. Test backward compatibility stats
    res = client.get("/api/live/stats")
    assert res.status_code == 200

    # 6. Test camera upsert API
    test_cam = {
        "id": 88,
        "name": "CAM 88: Temporary",
        "location": "Gate 4",
        "url": "http://192.168.1.88:4747/video",
        "enabled": True,
    }
    res = client.post("/api/cameras", json=test_cam)
    assert res.status_code == 200
    assert res.json()["camera"]["id"] == 88

    # 7. Test camera delete API
    res = client.delete("/api/cameras/88")
    assert res.status_code == 200
    assert res.json()["status"] == "deleted"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

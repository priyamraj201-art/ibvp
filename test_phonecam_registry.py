"""
Unit tests for the phone camera pairing/device registry.
"""

import os
import sys
import time
import base64
import pytest
from concurrent.futures import ThreadPoolExecutor

_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from dashboard.phonecam import (
    PhoneDeviceRegistry,
    get_lan_ip,
    generate_qr_png_base64,
)
from dashboard.stream_server import PhoneCamCapture, MultiCameraManager
from dashboard.camera_manager import CameraManager


def test_get_lan_ip_returns_a_nonempty_string():
    ip = get_lan_ip()
    assert isinstance(ip, str)
    assert len(ip) > 0


def test_generate_qr_png_base64_returns_decodable_png():
    b64 = generate_qr_png_base64("http://192.168.1.10:8000/phonecam/join/abc123")
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_pair_token_is_valid_until_expiry():
    registry = PhoneDeviceRegistry()
    token = registry.create_pair_token()
    assert registry.is_token_valid(token) is True
    assert registry.is_token_valid("not-a-real-token") is False


def test_pair_token_expires():
    registry = PhoneDeviceRegistry()
    registry.PAIR_TOKEN_TTL_SECONDS = 0.05
    token = registry.create_pair_token()
    time.sleep(0.1)
    assert registry.is_token_valid(token) is False


def test_get_or_create_capture_returns_same_instance_for_same_device():
    registry = PhoneDeviceRegistry()
    cap1 = registry.get_or_create_capture("device-a")
    cap2 = registry.get_or_create_capture("device-a")
    assert cap1 is cap2
    assert isinstance(cap1, PhoneCamCapture)


def test_get_or_create_capture_returns_different_instances_for_different_devices():
    registry = PhoneDeviceRegistry()
    cap1 = registry.get_or_create_capture("device-b")
    cap2 = registry.get_or_create_capture("device-c")
    assert cap1 is not cap2


def test_register_device_first_time_returns_int_cam_id(tmp_path, monkeypatch):
    """Test first-time device registration with isolated CameraManager and MultiCameraManager."""
    # Create isolated instances
    cfg_file = str(tmp_path / "test_cameras.json")
    isolated_camera_registry = CameraManager(config_path=cfg_file)
    isolated_multi_camera_manager = MultiCameraManager()

    # Monkeypatch the module-level singletons
    import dashboard.phonecam
    import dashboard.stream_server
    monkeypatch.setattr(dashboard.phonecam, "CAMERA_REGISTRY", isolated_camera_registry)
    monkeypatch.setattr(dashboard.phonecam, "MULTI_CAMERA_MANAGER", isolated_multi_camera_manager)
    monkeypatch.setattr(dashboard.stream_server, "CAMERA_REGISTRY", isolated_camera_registry)

    # Now test with isolated registry
    registry = PhoneDeviceRegistry()
    cam_id = registry.register_device("test-device-1", "Test Camera 1")
    assert isinstance(cam_id, int)
    assert cam_id > 0


def test_register_device_reconnect_returns_same_cam_id(tmp_path, monkeypatch):
    """Test reconnect scenario with isolated CameraManager and MultiCameraManager."""
    # Create isolated instances
    cfg_file = str(tmp_path / "test_cameras.json")
    isolated_camera_registry = CameraManager(config_path=cfg_file)
    isolated_multi_camera_manager = MultiCameraManager()

    # Monkeypatch the module-level singletons
    import dashboard.phonecam
    import dashboard.stream_server
    monkeypatch.setattr(dashboard.phonecam, "CAMERA_REGISTRY", isolated_camera_registry)
    monkeypatch.setattr(dashboard.phonecam, "MULTI_CAMERA_MANAGER", isolated_multi_camera_manager)
    monkeypatch.setattr(dashboard.stream_server, "CAMERA_REGISTRY", isolated_camera_registry)

    # Test with isolated registry
    registry = PhoneDeviceRegistry()
    cam_id_1 = registry.register_device("test-device-2", "Test Camera 2")
    # Count cameras before reconnect
    initial_camera_count = len(isolated_camera_registry.load_cameras())
    # Call again with same device_id (reconnect scenario)
    cam_id_2 = registry.register_device("test-device-2", "Test Camera 2 Updated")
    # Count cameras after reconnect
    final_camera_count = len(isolated_camera_registry.load_cameras())
    # Should return same cam_id
    assert cam_id_1 == cam_id_2
    # Should NOT have created a new camera entry (count should be the same)
    assert initial_camera_count == final_camera_count


def test_register_device_concurrent_same_device_no_race(tmp_path, monkeypatch):
    """Test concurrency with isolated CameraManager and MultiCameraManager."""
    # Create isolated instances
    cfg_file = str(tmp_path / "test_cameras.json")
    isolated_camera_registry = CameraManager(config_path=cfg_file)
    isolated_multi_camera_manager = MultiCameraManager()

    # Monkeypatch the module-level singletons
    import dashboard.phonecam
    import dashboard.stream_server
    monkeypatch.setattr(dashboard.phonecam, "CAMERA_REGISTRY", isolated_camera_registry)
    monkeypatch.setattr(dashboard.phonecam, "MULTI_CAMERA_MANAGER", isolated_multi_camera_manager)
    monkeypatch.setattr(dashboard.stream_server, "CAMERA_REGISTRY", isolated_camera_registry)

    # Test concurrency with isolated registry
    registry = PhoneDeviceRegistry()
    device_id = "concurrent-test-device"
    results = []

    def register_in_thread():
        cam_id = registry.register_device(device_id, "Concurrent Test")
        results.append(cam_id)

    # Spin up multiple threads all registering the same device_id
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(register_in_thread) for _ in range(10)]
        for future in futures:
            future.result()

    # All threads should have gotten the same cam_id
    assert len(set(results)) == 1, f"Expected all threads to return same cam_id, got {results}"
    # There should be exactly one entry in _device_cam_ids for this device
    assert device_id in registry._device_cam_ids
    cam_id = registry._device_cam_ids[device_id]
    # Find this camera in the registry to verify it exists (and is the only one for this device)
    all_cameras = isolated_camera_registry.load_cameras()
    matching_cameras = [c for c in all_cameras if c.get("id") == cam_id]
    assert len(matching_cameras) == 1, f"Expected exactly one camera with cam_id={cam_id}, got {matching_cameras}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

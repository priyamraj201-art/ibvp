"""
Unit tests for the phone camera pairing/device registry.
"""

import os
import sys
import time
import base64
import pytest

_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from dashboard.phonecam import (
    PhoneDeviceRegistry,
    get_lan_ip,
    generate_qr_png_base64,
)
from dashboard.stream_server import PhoneCamCapture


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


if __name__ == "__main__":
    pytest.main(["-v", __file__])

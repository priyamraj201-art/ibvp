"""
Phone Camera Pairing & Device Registry for the "Shared Perception" feature.

Handles:
  - QR-code pairing tokens (short-lived, LAN-scoped join guard)
  - Per-device PhoneCamCapture lookup
  - Auto-registering a phone as a camera in the existing CAMERA_REGISTRY
    and starting its pipeline worker via MULTI_CAMERA_MANAGER
"""

import base64
import io
import os
import secrets
import socket
import sys
import threading
import time
from typing import Dict, Optional

import qrcode
from loguru import logger

_bytetrack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from dashboard.camera_manager import CAMERA_REGISTRY
from dashboard.stream_server import PhoneCamCapture, MULTI_CAMERA_MANAGER


def get_lan_ip() -> str:
    """Best-effort detection of this machine's LAN-facing IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def generate_qr_png_base64(data: str) -> str:
    """Render `data` as a QR code PNG, returned as a base64-encoded string."""
    img = qrcode.make(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class PhoneDeviceRegistry:
    """Thread-safe registry of pairing tokens and connected phone-camera devices."""

    PAIR_TOKEN_TTL_SECONDS = 600  # 10 minutes

    def __init__(self):
        self._lock = threading.RLock()
        self._pair_tokens: Dict[str, float] = {}  # token -> expiry epoch
        self._captures: Dict[str, PhoneCamCapture] = {}  # device_id -> capture
        self._device_cam_ids: Dict[str, int] = {}  # device_id -> cam_id

    def create_pair_token(self) -> str:
        with self._lock:
            token = secrets.token_urlsafe(6)
            self._pair_tokens[token] = time.time() + self.PAIR_TOKEN_TTL_SECONDS
            return token

    def is_token_valid(self, token: str) -> bool:
        with self._lock:
            expiry = self._pair_tokens.get(token)
            return expiry is not None and time.time() < expiry

    def get_or_create_capture(self, device_id: str) -> PhoneCamCapture:
        with self._lock:
            if device_id not in self._captures:
                self._captures[device_id] = PhoneCamCapture(device_id)
            return self._captures[device_id]

    def register_device(self, device_id: str, name: str) -> int:
        """Ensure a camera registry entry + running worker exist for this device.

        Returns the cam_id (existing one on reconnect, newly assigned otherwise).
        """
        with self._lock:
            existing_cam_id = self._device_cam_ids.get(device_id)
            if existing_cam_id is not None:
                return existing_cam_id

            cam = CAMERA_REGISTRY.upsert_camera({
                "name": name or "Phone Camera",
                "location": "Phone (Shared Perception)",
                "url": f"phonecam://{device_id}",
                "enabled": True,
            })
            cam_id = cam["id"]
            self._device_cam_ids[device_id] = cam_id

        logger.info(f"[PhoneCam] Registered device {device_id} as camera {cam_id} ({name})")
        MULTI_CAMERA_MANAGER.start_camera(cam_id)
        return cam_id


PHONE_DEVICE_REGISTRY = PhoneDeviceRegistry()

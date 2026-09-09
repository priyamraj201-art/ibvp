"""
Camera Configuration Manager for IBVAP Surveillance System.
Handles loading, saving, and updating the camera registry (cameras.json).
"""

import os
import json
import threading
from typing import List, Dict, Any, Optional
from loguru import logger

_BYTETRACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CAMERAS_FILE = os.path.join(_BYTETRACK_ROOT, "cameras.json")

DEFAULT_CAMERAS: List[Dict[str, Any]] = [
    {
        "id": 0,
        "name": "CAM 01: Checkpost Alpha",
        "location": "Main Entrance",
        "url": "http://192.168.137.15:4747/video",
        "enabled": True,
    },
    {
        "id": 1,
        "name": "CAM 02: North Fence Line",
        "location": "Sector 3 Perimeter",
        "url": "http://192.168.137.22:4747/video",
        "enabled": True,
    },
    {
        "id": 2,
        "name": "CAM 03: Laptop / USB Webcam",
        "location": "Control Room",
        "url": "0",
        "enabled": True,
    },
    {
        "id": 3,
        "name": "CAM 04: Border Highway",
        "location": "Approach Road",
        "url": "videos/palace.mp4",
        "enabled": True,
    },
]


class CameraManager:
    """Thread-safe persistent camera registry manager."""

    def __init__(self, config_path: str = DEFAULT_CAMERAS_FILE):
        self.config_path = os.path.abspath(config_path)
        self._lock = threading.RLock()
        self._ensure_config_exists()

    def _ensure_config_exists(self):
        """Create cameras.json with defaults if missing or empty."""
        with self._lock:
            if not os.path.isfile(self.config_path) or os.path.getsize(self.config_path) == 0:
                try:
                    os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                    with open(self.config_path, "w", encoding="utf-8") as f:
                        json.dump(DEFAULT_CAMERAS, f, indent=2)
                    logger.info(f"[CameraManager] Initialized default cameras at {self.config_path}")
                except Exception as e:
                    logger.error(f"[CameraManager] Failed to create cameras.json: {e}")

    def load_cameras(self) -> List[Dict[str, Any]]:
        """Load list of camera configurations."""
        with self._lock:
            if not os.path.isfile(self.config_path):
                return [dict(c) for c in DEFAULT_CAMERAS]

            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    logger.warning("[CameraManager] cameras.json did not contain a list. Falling back.")
                    return [dict(c) for c in DEFAULT_CAMERAS]
            except Exception as e:
                logger.error(f"[CameraManager] Error reading cameras.json: {e}")
                return [dict(c) for c in DEFAULT_CAMERAS]

    def save_cameras(self, cameras: List[Dict[str, Any]]) -> bool:
        """Persist cameras list to cameras.json."""
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(cameras, f, indent=2)
                return True
            except Exception as e:
                logger.error(f"[CameraManager] Error saving cameras.json: {e}")
                return False

    def get_camera(self, cam_id: int) -> Optional[Dict[str, Any]]:
        """Find a camera by its numeric ID."""
        cams = self.load_cameras()
        for c in cams:
            if c.get("id") == cam_id:
                return dict(c)
        return None

    def upsert_camera(self, cam_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new camera or update an existing one by ID.
        If 'id' is not provided, assigns the next available positive integer.
        """
        cams = self.load_cameras()
        raw_id = cam_data.get("id")

        if raw_id is None or raw_id == "" or raw_id == -1:
            # Generate next available ID
            existing_ids = [c.get("id", 0) for c in cams if isinstance(c.get("id"), int)]
            cam_id = max(existing_ids) + 1 if existing_ids else 0
        else:
            cam_id = int(raw_id)

        clean_cam = {
            "id": cam_id,
            "name": str(cam_data.get("name", f"CAM {cam_id:02d}")).strip(),
            "location": str(cam_data.get("location", "Sector Alpha")).strip(),
            "url": str(cam_data.get("url", "0")).strip(),
            "enabled": bool(cam_data.get("enabled", True)),
        }

        updated = False
        for i, c in enumerate(cams):
            if c.get("id") == cam_id:
                cams[i] = clean_cam
                updated = True
                break

        if not updated:
            cams.append(clean_cam)

        # Sort by ID ascending
        cams.sort(key=lambda x: x.get("id", 0))
        self.save_cameras(cams)
        return clean_cam

    def delete_camera(self, cam_id: int) -> bool:
        """Remove a camera by ID."""
        cams = self.load_cameras()
        new_cams = [c for c in cams if c.get("id") != cam_id]
        if len(new_cams) != len(cams):
            return self.save_cameras(new_cams)
        return False

    def toggle_camera(self, cam_id: int, enabled: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """Toggle or set the enabled status of a camera."""
        cams = self.load_cameras()
        target = None
        for c in cams:
            if c.get("id") == cam_id:
                if enabled is None:
                    c["enabled"] = not c.get("enabled", True)
                else:
                    c["enabled"] = bool(enabled)
                target = dict(c)
                break

        if target:
            self.save_cameras(cams)
        return target

    def get_enabled_cameras(self) -> List[Dict[str, Any]]:
        """Return only cameras where enabled is True."""
        return [c for c in self.load_cameras() if c.get("enabled", True)]


# Global singleton instance
CAMERA_REGISTRY = CameraManager()

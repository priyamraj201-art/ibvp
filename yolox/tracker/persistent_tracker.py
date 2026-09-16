"""
Enterprise Persistent Tracker & Cross-Camera Entity Manager.
Assigns persistent IDs (PER-xxxx for humans, VEH-xxxx for vehicles) to all tracked targets,
tracks camera occurrences, updates identification states (FRS/ANPR), and maintains
forensic snapshots and sighting audit trails.
"""

import os
import time
import queue
import threading
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from .entity_db import EntityDatabase


class PersistentTracker:
    """
    Global Persistent Entity Tracking and Re-Identification Engine.
    Coordinates across multiple cameras to ensure every tracked target receives
    a persistent ID and records all camera appearances and biometric/OCR statuses.
    """

    def __init__(
        self,
        db_path: str = "tracked_entities.db",
        crops_dir: str = "assets/entity_crops",
        sighting_interval_sec: float = 0.5,
    ):
        self.db_path = db_path
        self.crops_dir = os.path.abspath(crops_dir)
        os.makedirs(self.crops_dir, exist_ok=True)
        self.sighting_interval_sec = sighting_interval_sec

        self.db = EntityDatabase(db_path=self.db_path)

        # Mapping: (cam_id, local_track_id) -> persistent_id
        self._local_to_persistent: Dict[Tuple[int, int], str] = {}
        # Mapping: (cam_id, local_track_id) -> last_seen_time
        self._local_last_seen: Dict[Tuple[int, int], float] = {}
        # Throttling: persistent_id -> last_recorded_time
        self._last_sighting_record: Dict[str, float] = {}

        self._lock = threading.Lock()

        # Non-blocking async queue for database writes
        self._db_queue = queue.Queue(maxsize=1000)
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._db_worker, name="PersistentTracker-DBWorker", daemon=True
        )
        self._worker_thread.start()

        logger.info(f"PersistentTracker initialized (DB: {self.db_path}, Crops: {self.crops_dir})")

    def _db_worker(self):
        """Background worker thread draining database writes."""
        while self._running:
            try:
                task = self._db_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if task is None:
                break

            try:
                op, kwargs = task
                if op == "upsert_entity":
                    self.db.upsert_entity(**kwargs)
                elif op == "record_sighting":
                    self.db.record_sighting(**kwargs)
            except Exception as e:
                logger.error(f"[PersistentTracker] DB write worker error: {e}")
            finally:
                self._db_queue.task_done()

    def _queue_write(self, op: str, kwargs: dict):
        try:
            self._db_queue.put_nowait((op, kwargs))
        except queue.Full:
            pass

    def assign_or_get_id(
        self,
        cam_id: int,
        cam_name: str,
        local_track_id: int,
        entity_type: str,
        tlwh: List[float],
        frame: Optional[np.ndarray] = None,
        speed: float = 0.0,
        now_ts: Optional[float] = None,
    ) -> str:
        """
        Assign or retrieve the persistent ID for a tracked target on a specific camera.
        Immediately allocates a new persistent ID if not seen before.
        """
        now = now_ts or time.time()
        key = (cam_id, local_track_id)
        norm_type = "HUMAN" if entity_type.upper() in ("HUMAN", "PERSON") else ("VEHICLE" if entity_type.upper() in ("VEHICLE", "CAR") else "OBJECT")

        with self._lock:
            if key in self._local_to_persistent:
                pid = self._local_to_persistent[key]
                self._local_last_seen[key] = now
                should_record = (now - self._last_sighting_record.get(pid, 0.0)) >= self.sighting_interval_sec
                if should_record:
                    self._last_sighting_record[pid] = now
                    # Queue periodic update
                    bbox_str = f"{int(tlwh[0])},{int(tlwh[1])},{int(tlwh[2])},{int(tlwh[3])}" if len(tlwh) >= 4 else ""
                    self._queue_write("upsert_entity", {
                        "persistent_id": pid,
                        "entity_type": norm_type,
                        "first_seen": now,
                        "last_seen": now,
                        "cameras_seen": [cam_name],
                        "last_camera_id": cam_id,
                        "last_camera_name": cam_name,
                    })
                    self._queue_write("record_sighting", {
                        "persistent_id": pid,
                        "camera_id": cam_id,
                        "camera_name": cam_name,
                        "timestamp": now,
                        "bbox": bbox_str,
                        "local_track_id": local_track_id,
                        "speed": speed,
                    })
                return pid

            # New local tracklet on this camera: mint a persistent ID
            pid = self.db.get_next_id(norm_type)
            self._local_to_persistent[key] = pid
            self._local_last_seen[key] = now
            self._last_sighting_record[pid] = now

        # Save forensic thumbnail crop on first observation
        thumb_rel_path = ""
        if frame is not None and len(tlwh) >= 4:
            try:
                x1, y1, w, h = map(int, tlwh)
                fh, fw = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x1 + w), min(fh, y1 + h)
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    crop = frame[y1:y2, x1:x2]
                    thumb_filename = f"{pid}_{int(now)}.jpg"
                    thumb_abs_path = os.path.join(self.crops_dir, thumb_filename)
                    cv2.imwrite(thumb_abs_path, crop)
                    thumb_rel_path = f"assets/entity_crops/{thumb_filename}"
            except Exception as e:
                logger.debug(f"Error saving initial thumbnail crop for {pid}: {e}")

        bbox_str = f"{int(tlwh[0])},{int(tlwh[1])},{int(tlwh[2])},{int(tlwh[3])}" if len(tlwh) >= 4 else ""
        default_name = f"Unidentified {norm_type.title()}"

        # Register immediately in DB
        self.db.upsert_entity(
            persistent_id=pid,
            entity_type=norm_type,
            first_seen=now,
            last_seen=now,
            cameras_seen=[cam_name],
            last_camera_id=cam_id,
            last_camera_name=cam_name,
            is_identified=False,
            identified_id="UNIDENTIFIED",
            identified_name=default_name,
            category="NORMAL",
            confidence=0.0,
            thumbnail_path=thumb_rel_path,
        )

        self.db.record_sighting(
            persistent_id=pid,
            camera_id=cam_id,
            camera_name=cam_name,
            timestamp=now,
            bbox=bbox_str,
            local_track_id=local_track_id,
            is_identified=False,
            identified_id="UNIDENTIFIED",
            confidence=0.0,
            speed=speed,
        )

        logger.info(f"🆕 [Persistent Entity Assigned] {pid} ({norm_type}) on {cam_name} (Local Track #{local_track_id})")
        return pid

    def update_identification(
        self,
        cam_id: int,
        local_track_id: int,
        is_identified: bool,
        identified_id: str,
        identified_name: str,
        category: str = "NORMAL",
        confidence: float = 0.0,
        embedding: Optional[np.ndarray] = None,
        face_crop: Optional[np.ndarray] = None,
        cam_name: str = "",
    ) -> str:
        """
        Update biometric (FRS) or OCR (ANPR) identification status for an active track.
        Handles cross-camera correlation if the identified target was already seen on another camera.
        """
        key = (cam_id, local_track_id)
        with self._lock:
            curr_pid = self._local_to_persistent.get(key)
        if not curr_pid:
            return ""

        now = time.time()
        final_pid = curr_pid

        # Cross-camera correlation: check if this person_id or plate_number already exists
        if is_identified and identified_id and identified_id != "UNIDENTIFIED":
            existing = self.db.find_by_identified_id(identified_id)
            if existing and existing["persistent_id"] != curr_pid:
                canonical_pid = existing["persistent_id"]
                logger.info(
                    f"🔗 [Cross-Camera Match] Local track #{local_track_id} on Cam {cam_id} ({curr_pid}) "
                    f"matched existing identity {canonical_pid} ({identified_name} - {identified_id})"
                )
                with self._lock:
                    self._local_to_persistent[key] = canonical_pid
                final_pid = canonical_pid

        # Optional updated face crop
        thumb_rel_path = ""
        if face_crop is not None and face_crop.size > 0:
            try:
                thumb_filename = f"{final_pid}_face_{int(now)}.jpg"
                thumb_abs_path = os.path.join(self.crops_dir, thumb_filename)
                cv2.imwrite(thumb_abs_path, face_crop)
                thumb_rel_path = f"assets/entity_crops/{thumb_filename}"
            except Exception as e:
                logger.debug(f"Error saving face crop for {final_pid}: {e}")

        # Update entity in DB
        entity_type = "HUMAN" if "PER-" in final_pid else ("VEHICLE" if "VEH-" in final_pid else "OBJECT")
        cams = [cam_name] if cam_name else []

        self._queue_write("upsert_entity", {
            "persistent_id": final_pid,
            "entity_type": entity_type,
            "first_seen": now,
            "last_seen": now,
            "cameras_seen": cams,
            "last_camera_id": cam_id,
            "last_camera_name": cam_name,
            "is_identified": is_identified,
            "identified_id": identified_id if is_identified else "UNIDENTIFIED",
            "identified_name": identified_name if is_identified else f"Unidentified {entity_type.title()}",
            "category": category,
            "confidence": confidence,
            "thumbnail_path": thumb_rel_path,
            "embedding": embedding,
        })

        self._queue_write("record_sighting", {
            "persistent_id": final_pid,
            "camera_id": cam_id,
            "camera_name": cam_name,
            "timestamp": now,
            "bbox": "",
            "local_track_id": local_track_id,
            "is_identified": is_identified,
            "identified_id": identified_id if is_identified else "",
            "confidence": confidence,
            "speed": 0.0,
        })

        return final_pid

    def get_persistent_id(self, cam_id: int, local_track_id: int) -> Optional[str]:
        """Look up active persistent ID for a camera's local tracklet."""
        with self._lock:
            return self._local_to_persistent.get((cam_id, local_track_id))

    def cleanup_inactive_tracks(self, timeout_sec: float = 60.0):
        """Evict stale in-memory camera mappings that haven't been observed recently."""
        now = time.time()
        with self._lock:
            stale_keys = [k for k, last_t in self._local_last_seen.items() if (now - last_t) > timeout_sec]
            for k in stale_keys:
                self._local_to_persistent.pop(k, None)
                self._local_last_seen.pop(k, None)

    def close(self):
        """Stop background worker and flush queue."""
        self._running = False
        try:
            self._db_queue.put_nowait(None)
        except Exception:
            pass


# Global singleton instance for use across application and multi-camera streams
GLOBAL_PERSISTENT_TRACKER = PersistentTracker()

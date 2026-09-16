"""
Asynchronous Person Re-ID Pipeline.
Coordinates non-blocking appearance feature extraction, multi-camera Re-ID matching,
and retained ID binding to maintain 30 FPS across multi-camera video streams.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import queue
import threading
from typing import Optional, Dict, Tuple, List, Callable, Any
import numpy as np
from loguru import logger

from .reid_extractor import PersonReIDExtractor
from .suspect_registry import PersonReIDRegistry, RetainedPerson, GLOBAL_REID_REGISTRY


class ReIDPipeline:
    """
    High-Throughput Asynchronous Re-ID Pipeline.
    Runs Re-ID inference in a dedicated background worker thread so the main video
    pipeline never drops frames waiting on OSNet inference or database operations.
    """

    def __init__(
        self,
        extractor: Optional[PersonReIDExtractor] = None,
        registry: Optional[PersonReIDRegistry] = None,
        match_threshold: float = 0.70,
        max_queue_size: int = 300,
    ):
        self.extractor = extractor or PersonReIDExtractor()
        self.registry = registry or GLOBAL_REID_REGISTRY
        self.match_threshold = match_threshold

        # Active binding: (cam_id, local_track_id) -> retained_id
        self._lock = threading.RLock()
        self._track_to_retained: Dict[Tuple[int, int], str] = {}
        # Reverse mapping: retained_id -> (cam_id, local_track_id, last_seen)
        self._retained_to_track: Dict[str, Tuple[int, int, float]] = {}

        # Alert/event listener callbacks
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

        # Work queue and worker thread
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="ReIDPipeline-Worker", daemon=True
        )
        self._worker_thread.start()

        logger.info("[ReIDPipeline] Asynchronous Retained ID Re-ID worker initialized.")

    def register_listener(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback invoked when a person is registered or re-identified."""
        with self._lock:
            self._listeners.append(callback)

    # Backward compatibility alias
    register_alert_listener = register_listener

    def _broadcast_event(self, payload: Dict[str, Any]):
        """Notify all registered listeners of a Re-ID event."""
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(payload)
            except Exception as e:
                logger.debug(f"Error in ReID listener: {e}")

    # Backward compatibility alias
    _broadcast_alert = _broadcast_event

    def get_retained_id(self, cam_id: int, local_track_id: int) -> Optional[str]:
        """Look up active retained ID for a local track."""
        with self._lock:
            return self._track_to_retained.get((cam_id, local_track_id))

    def bind_retained_id(self, cam_id: int, local_track_id: int, retained_id: str):
        """Bind a retained person ID to a local camera track."""
        with self._lock:
            self._track_to_retained[(cam_id, local_track_id)] = retained_id
            self._retained_to_track[retained_id] = (cam_id, local_track_id, time.time())

    def unbind_stale_tracks(self, max_age_seconds: float = 60.0):
        """Clean up local track bindings that haven't been refreshed."""
        now = time.time()
        with self._lock:
            stale_keys = []
            for k, pid in self._track_to_retained.items():
                entry = self._retained_to_track.get(pid)
                if entry and (now - entry[2] > max_age_seconds):
                    stale_keys.append(k)
            for k in stale_keys:
                pid = self._track_to_retained.pop(k, None)
                if pid:
                    self._retained_to_track.pop(pid, None)

    def submit_observation(
        self,
        cam_id: int,
        cam_name: str,
        local_track_id: int,
        crop: np.ndarray,
        bbox: List[int],
        is_breach: bool = False,
        zone_name: str = "",
    ):
        """
        Submit a candidate person crop for asynchronous Re-ID processing.
        Non-blocking: drops request if queue is completely full under heavy load.
        """
        if crop is None or crop.size == 0:
            return

        task = {
            "cam_id": cam_id,
            "cam_name": cam_name,
            "local_track_id": local_track_id,
            "crop": crop.copy(),
            "bbox": bbox,
            "timestamp": time.time(),
        }

        try:
            self._queue.put_nowait(task)
        except queue.Full:
            pass

    def _worker_loop(self):
        """Continuous background worker processing Re-ID appearance extraction and matching."""
        while self._running:
            try:
                task = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._process_task(task)
            except Exception as e:
                logger.error(f"[ReIDPipeline] Error processing task: {e}")
            finally:
                self._queue.task_done()

    def _process_task(self, task: dict):
        cam_id = task["cam_id"]
        cam_name = task["cam_name"]
        tid = task["local_track_id"]
        crop = task["crop"]
        bbox = task["bbox"]
        now = task["timestamp"]

        # 1. Extract 512-D L2-normalized Re-ID embedding
        embedding = self.extractor.extract_feature(crop)

        # 2. Check if track already has a retained ID
        existing_pid = self.get_retained_id(cam_id, tid)

        if existing_pid:
            # Already bound: update sighting history and progressive appearance
            self.registry.update_person(
                retained_id=existing_pid,
                camera_id=cam_id,
                camera_name=cam_name,
                crop=crop,
                embedding=embedding,
                bbox=bbox,
            )
            with self._lock:
                self._retained_to_track[existing_pid] = (cam_id, tid, now)
            return

        # 3. Newly observed track: match against active Re-ID gallery
        matched_pid, score = self.registry.match_person(embedding, threshold=self.match_threshold)

        if matched_pid:
            # RE-ID MATCH: Re-bind retained ID to this tracklet
            self.bind_retained_id(cam_id, tid, matched_pid)
            self.registry.update_person(
                retained_id=matched_pid,
                camera_id=cam_id,
                camera_name=cam_name,
                crop=crop,
                embedding=embedding,
                bbox=bbox,
            )
            logger.info(
                f"🎯 [RE-ID MATCH ({score:.2f})] Retained {matched_pid} for track #{tid} on {cam_name}"
            )
            self._broadcast_event({
                "type": "REID_MATCH",
                "retained_id": matched_pid,
                "persistent_id": matched_pid,
                "camera_id": cam_id,
                "camera_name": cam_name,
                "track_id": tid,
                "score": round(score, 3),
                "timestamp": time.strftime("%H:%M:%S"),
            })
        else:
            # NEW PERSON: Mint clean retained ID
            new_pid = self.registry.register_person(
                camera_id=cam_id,
                camera_name=cam_name,
                crop=crop,
                embedding=embedding,
                bbox=bbox,
            )
            self.bind_retained_id(cam_id, tid, new_pid)
            logger.info(
                f"✨ [RE-ID NEW] Minted {new_pid} for track #{tid} on {cam_name}"
            )
            self._broadcast_event({
                "type": "REID_NEW",
                "retained_id": new_pid,
                "persistent_id": new_pid,
                "camera_id": cam_id,
                "camera_name": cam_name,
                "track_id": tid,
                "score": 1.0,
                "timestamp": time.strftime("%H:%M:%S"),
            })

    def stop(self):
        """Stop background worker thread cleanly."""
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        logger.info("[ReIDPipeline] Worker stopped cleanly.")


# Global singleton instance
GLOBAL_REID_PIPELINE = ReIDPipeline()

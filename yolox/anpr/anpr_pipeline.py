import os
import time
import queue
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from loguru import logger

from .watchlist_db import WatchlistDB
from .plate_detector import PlateDetector
from .ocr_engine import OCREngine


class TrackCandidate:
    """
    Maintains observation quality and recognition state for a tracked vehicle/target.
    """
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.best_score: float = 0.0
        self.best_area: float = 0.0
        self.best_crop: Optional[np.ndarray] = None
        self.status: str = "UNPROCESSED"  # UNPROCESSED, QUEUED, COMPLETED, NO_PLATE
        self.last_update_time: float = time.time()
        self.last_attempt_time: float = 0.0
        self.plate_result: Optional[Dict[str, Any]] = None


class ANPRPipeline:
    """
    High-performance Asynchronous ANPR Controller.
    
    1. Selects the best-quality crop per track (largest area + highest confidence).
    2. Runs OCR asynchronously and caches results by track_id.
    3. Periodically re-evaluates unconfirmed tracks when new observations arrive.
    4. Automatically performs SQLite security watchlist queries and detection logging.
    """

    def __init__(
        self,
        db_path: str = "anpr_watchlist.db",
        min_box_area: float = 400.0,
        quality_boost_ratio: float = 1.25,
        recheck_interval_sec: float = 0.6,
        num_workers: int = 1,
        save_crops_dir: Optional[str] = None,
    ):
        self.min_box_area = min_box_area
        self.quality_boost_ratio = quality_boost_ratio
        self.recheck_interval_sec = recheck_interval_sec
        self.num_workers = num_workers
        self.save_crops_dir = save_crops_dir

        if self.save_crops_dir:
            os.makedirs(self.save_crops_dir, exist_ok=True)

        # Core Components
        self.watchlist_db = WatchlistDB(db_path=db_path)
        self.plate_detector = PlateDetector()
        self.ocr_engine = OCREngine()

        # Track Candidates & Results Cache (Thread-Safe)
        self.candidates: Dict[int, TrackCandidate] = {}
        self.results_cache: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Work Queue & Thread Pool
        self.task_queue = queue.Queue(maxsize=128)
        self.is_running = True
        self.workers = []
        self._start_workers()

        logger.info(f"ANPR Pipeline initialized (Workers: {self.num_workers}, Min Area: {self.min_box_area})")

    def _start_workers(self):
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"ANPR-Worker-{i}", daemon=True)
            t.start()
            self.workers.append(t)

    def _worker_loop(self):
        """
        Background worker that processes vehicle crops asynchronously.
        """
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None:
                break

            track_id, vehicle_crop, bbox_area, conf, timestamp = task
            try:
                # 1. Plate Extraction
                plate_crop, plate_box = self.plate_detector.extract_plate(vehicle_crop)

                # 2. Run OCR Inference
                plate_text, ocr_conf = self.ocr_engine.recognize_plate(plate_crop)

                # If no text found on cropped sub-region, attempt full vehicle crop
                if not plate_text and plate_crop is not vehicle_crop:
                    plate_text, ocr_conf = self.ocr_engine.recognize_plate(vehicle_crop)

                if plate_text:
                    # 3. Query Watchlist
                    watchlist_hit = self.watchlist_db.lookup_plate(plate_text)
                    is_flagged = watchlist_hit is not None
                    alert_category = watchlist_hit["alert_category"] if is_flagged else "NORMAL"
                    owner = watchlist_hit["owner_name"] if is_flagged else ""
                    notes = watchlist_hit["notes"] if is_flagged else ""

                    # 4. Optional Crop Saving for Audit
                    crop_file = None
                    if self.save_crops_dir and plate_crop is not None and plate_crop.size > 0:
                        import cv2
                        crop_file = os.path.join(self.save_crops_dir, f"{plate_text}_{track_id}_{int(timestamp)}.jpg")
                        cv2.imwrite(crop_file, plate_crop)

                    # 5. Log to SQLite
                    self.watchlist_db.log_detection(
                        track_id=track_id,
                        plate_number=plate_text,
                        confidence=ocr_conf,
                        is_flagged=is_flagged,
                        alert_category=alert_category if is_flagged else None,
                        bbox_area=bbox_area,
                        crop_path=crop_file,
                    )

                    # 6. Update Cache
                    plate_data = {
                        "plate_number": plate_text,
                        "confidence": ocr_conf,
                        "is_flagged": is_flagged,
                        "alert_category": alert_category,
                        "owner_name": owner,
                        "notes": notes,
                        "timestamp": timestamp,
                        "plate_box": plate_box,
                    }

                    with self._lock:
                        self.results_cache[track_id] = plate_data
                        if track_id in self.candidates:
                            self.candidates[track_id].status = "COMPLETED"
                            self.candidates[track_id].plate_result = plate_data

                    if is_flagged:
                        logger.warning(
                            f"🚨 [WATCHLIST HIT] Track #{track_id} Plate: {plate_text} | Category: {alert_category} | Owner: {owner}"
                        )
                    else:
                        logger.info(f"🚗 [ANPR] Track #{track_id} Plate: {plate_text} (Conf: {ocr_conf:.2f})")
                else:
                    with self._lock:
                        if track_id in self.candidates:
                            self.candidates[track_id].status = "NO_PLATE"

            except Exception as e:
                logger.error(f"Error processing ANPR for track #{track_id}: {e}")
            finally:
                self.task_queue.task_done()

    def process_frame(
        self,
        raw_img: np.ndarray,
        tlwhs: List[List[float]],
        track_ids: List[int],
        scores: List[float],
        current_time: Optional[float] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate active tracks and schedule OCR without blocking the main tracking loop.
        """
        if current_time is None:
            current_time = time.time()

        h_img, w_img = raw_img.shape[:2]
        active_ids = set(track_ids)

        # Clean stale candidates older than 60 seconds
        with self._lock:
            stale_ids = [
                tid for tid, cand in self.candidates.items()
                if tid not in active_ids and current_time - cand.last_update_time > 60.0
            ]
            for tid in stale_ids:
                del self.candidates[tid]
                self.results_cache.pop(tid, None)

        # Evaluate candidate crops for each active track
        for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
            x, y, w, h = map(int, tlwh)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

            box_area = float((x2 - x1) * (y2 - y1))
            if box_area < self.min_box_area:
                continue

            quality_score = box_area * float(score)

            with self._lock:
                cand = self.candidates.get(track_id)
                if cand is None:
                    cand = TrackCandidate(track_id)
                    self.candidates[track_id] = cand

                cand.last_update_time = current_time

                # Decide if we should queue this crop
                should_queue = False
                if cand.status == "UNPROCESSED":
                    cand.best_score = quality_score
                    cand.best_area = box_area
                    cand.best_crop = raw_img[y1:y2, x1:x2].copy()
                    cand.last_attempt_time = current_time
                    should_queue = True
                elif (
                    cand.status == "NO_PLATE"
                    and (current_time - cand.last_attempt_time > self.recheck_interval_sec)
                ):
                    # Periodically retry unconfirmed tracks so held-up plates are detected
                    cand.best_score = quality_score
                    cand.best_area = box_area
                    cand.best_crop = raw_img[y1:y2, x1:x2].copy()
                    cand.last_attempt_time = current_time
                    should_queue = True
                elif (
                    cand.status in ("COMPLETED", "NO_PLATE")
                    and quality_score > cand.best_score * self.quality_boost_ratio
                ):
                    # Significantly better view of vehicle
                    cand.best_score = quality_score
                    cand.best_area = box_area
                    cand.best_crop = raw_img[y1:y2, x1:x2].copy()
                    cand.last_attempt_time = current_time
                    should_queue = True

                if should_queue and cand.best_crop is not None:
                    cand.status = "QUEUED"
                    try:
                        self.task_queue.put_nowait(
                            (track_id, cand.best_crop, box_area, float(score), current_time)
                        )
                    except queue.Full:
                        pass

        with self._lock:
            return dict(self.results_cache)

    def get_track_plate(self, track_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.results_cache.get(track_id)

    def stop(self):
        """Cleanly terminate background workers."""
        self.is_running = False
        for _ in range(self.num_workers):
            try:
                self.task_queue.put_nowait(None)
            except Exception:
                pass
        for t in self.workers:
            t.join(timeout=1.0)
        logger.info("ANPR Pipeline background workers stopped.")

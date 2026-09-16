"""
Persistent Person Re-ID Registry & Retained Identity Manager.
Stores 512-D OSNet appearance embeddings, multi-camera sighting timelines, and forensic snapshots
for tracked individuals across all surveillance cameras.
"""

import os
import time
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import cv2
from loguru import logger

_BYTETRACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class RetainedPerson:
    """Complete multi-camera forensic profile for a person with a retained identity."""
    retained_id: str
    first_seen_time: float
    last_seen_time: float
    camera_id: int
    camera_name: str = ""
    camera_history: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    snapshots: List[str] = field(default_factory=list)
    status: str = "ACTIVE"
    best_score: float = 1.0

    @property
    def persistent_id(self) -> str:
        return self.retained_id

    @property
    def dwell_seconds(self) -> float:
        return max(0.0, self.last_seen_time - self.first_seen_time)

    def get_centroid_embedding(self) -> np.ndarray:
        """Calculate the normalized mean embedding across all observations."""
        if not self.embeddings:
            return np.zeros((512,), dtype=np.float32)
        arr = np.mean(self.embeddings, axis=0)
        norm = np.linalg.norm(arr)
        if norm > 1e-6:
            arr = arr / norm
        return arr.astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary for dashboard API."""
        return {
            "retained_id": self.retained_id,
            "persistent_id": self.retained_id,
            "first_seen_time": self.first_seen_time,
            "first_seen_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.first_seen_time)),
            "last_seen_time": self.last_seen_time,
            "last_seen_str": time.strftime("%H:%M:%S", time.localtime(self.last_seen_time)),
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "camera_history": self.camera_history,
            "snapshots": self.snapshots,
            "dwell_seconds": round(self.dwell_seconds, 1),
            "status": self.status,
            "observation_count": len(self.embeddings),
        }


# Backward compatibility alias
IntrusionDossier = RetainedPerson


class PersonReIDRegistry:
    """
    Thread-safe in-memory + SQLite Persistent Person Re-ID Registry.
    Performs cosine similarity search over normalized OSNet embeddings and retains person identities.
    """

    def __init__(
        self,
        db_path: str = "reid_identities.db",
        crops_dir: str = "assets/reid_crops",
        similarity_threshold: float = 0.70,
    ):
        if not os.path.isabs(db_path):
            db_path = os.path.join(_BYTETRACK_ROOT, db_path)
        if not os.path.isabs(crops_dir):
            crops_dir = os.path.join(_BYTETRACK_ROOT, crops_dir)

        self.db_path = os.path.abspath(db_path)
        self.crops_dir = os.path.abspath(crops_dir)
        os.makedirs(self.crops_dir, exist_ok=True)
        self.similarity_threshold = similarity_threshold

        self._lock = threading.RLock()
        self._persons: Dict[str, RetainedPerson] = {}
        self._seq_counter = 0

        self._init_sqlite()
        self._load_active_persons()

    def _init_sqlite(self):
        """Initialize SQLite database for persistent person identities."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reid_persons (
                    retained_id TEXT PRIMARY KEY,
                    first_seen_time REAL,
                    last_seen_time REAL,
                    camera_id INTEGER,
                    camera_name TEXT,
                    camera_history TEXT,
                    embeddings_blob BLOB,
                    snapshots_json TEXT,
                    status TEXT
                )
            """)
            conn.commit()

    def _load_active_persons(self):
        """Restore person identities from SQLite into fast in-memory cache."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM reid_persons")
                    rows = cursor.fetchall()
                    for row in rows:
                        pid = row[0]
                        first_t = row[1]
                        last_t = row[2]
                        cam_id = row[3]
                        cam_name = row[4]
                        cam_hist = json.loads(row[5]) if row[5] else []
                        embs_bytes = row[6]
                        embs = []
                        if embs_bytes:
                            embs_arr = np.frombuffer(embs_bytes, dtype=np.float32)
                            if len(embs_arr) % 512 == 0:
                                embs = [embs_arr[i : i + 512] for i in range(0, len(embs_arr), 512)]
                        snaps = json.loads(row[7]) if row[7] else []
                        status = row[8]

                        person = RetainedPerson(
                            retained_id=pid,
                            first_seen_time=first_t,
                            last_seen_time=last_t,
                            camera_id=cam_id,
                            camera_name=cam_name,
                            camera_history=cam_hist,
                            embeddings=embs,
                            snapshots=snaps,
                            status=status,
                        )
                        self._persons[pid] = person

                    cursor.execute("SELECT COUNT(*) FROM reid_persons")
                    self._seq_counter = cursor.fetchone()[0]
                logger.info(f"[PersonReIDRegistry] Loaded {len(self._persons)} retained persons from {self.db_path}")
            except Exception as e:
                logger.error(f"[PersonReIDRegistry] Error loading persons: {e}")

    def _generate_id(self) -> str:
        """Mint a clean Retained Person ID: REID-001, REID-002, etc."""
        self._seq_counter += 1
        return f"REID-{self._seq_counter:03d}"

    def register_person(
        self,
        camera_id: int,
        camera_name: str = "",
        crop: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        bbox: Optional[List[int]] = None,
    ) -> str:
        """
        Register a new person in the Re-ID gallery.
        Generates clean Retained ID, stores initial 512-D embedding, and captures snapshot.
        """
        now = time.time()
        with self._lock:
            pid = self._generate_id()

            # Save forensic snapshot crop
            snapshots = []
            if crop is not None and crop.size > 0:
                try:
                    filename = f"{pid}_{int(now)}.jpg"
                    abs_path = os.path.join(self.crops_dir, filename)
                    cv2.imwrite(abs_path, crop)
                    snapshots.append(f"assets/reid_crops/{filename}")
                except Exception as e:
                    logger.debug(f"Snapshot write error: {e}")

            embs = []
            if embedding is not None and len(embedding) == 512:
                norm = np.linalg.norm(embedding)
                normed = embedding / norm if norm > 1e-6 else embedding
                embs.append(normed.astype(np.float32))

            history_entry = {
                "timestamp": now,
                "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                "camera_id": camera_id,
                "camera_name": camera_name or f"CAM {camera_id:02d}",
                "event": "FIRST_SEEN",
                "bbox": bbox or [],
            }

            person = RetainedPerson(
                retained_id=pid,
                first_seen_time=now,
                last_seen_time=now,
                camera_id=camera_id,
                camera_name=camera_name or f"CAM {camera_id:02d}",
                camera_history=[history_entry],
                embeddings=embs,
                snapshots=snapshots,
                status="ACTIVE",
            )
            self._persons[pid] = person

            # Persist to SQLite
            self._save_to_db(person)
            logger.info(f"✨ [RE-ID REGISTER] Assigned {pid} on {camera_name or f'CAM {camera_id:02d}'}")
            return pid

    # Backward compatibility alias
    def register_suspect(self, zone_name: str = "", **kwargs) -> str:
        return self.register_person(**kwargs)

    def match_person(
        self,
        query_embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Match incoming track appearance embedding against all active retained persons.
        Returns: (best_retained_id, best_cosine_score) if score >= threshold, else (None, best_score).
        """
        if query_embedding is None or len(query_embedding) != 512:
            return None, 0.0

        thresh = threshold if threshold is not None else self.similarity_threshold

        norm = np.linalg.norm(query_embedding)
        if norm < 1e-6:
            return None, 0.0
        q = (query_embedding / norm).astype(np.float32)

        with self._lock:
            if not self._persons:
                return None, 0.0

            best_pid = None
            best_score = -1.0

            for pid, person in self._persons.items():
                if person.status != "ACTIVE":
                    continue

                # Compare against centroid embedding
                centroid = person.get_centroid_embedding()
                score = float(np.dot(q, centroid))

                # Also test individual embeddings for max appearance similarity
                for emb in person.embeddings:
                    s = float(np.dot(q, emb))
                    if s > score:
                        score = s

                if score > best_score:
                    best_score = score
                    best_pid = pid

            if best_score >= thresh:
                return best_pid, best_score
            return None, max(0.0, best_score)

    # Backward compatibility alias
    def match_suspect(self, query_embedding: np.ndarray, threshold: Optional[float] = None) -> Tuple[Optional[str], float]:
        return self.match_person(query_embedding, threshold)

    def update_person(
        self,
        retained_id: str,
        camera_id: int,
        camera_name: str = "",
        crop: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        bbox: Optional[List[int]] = None,
        zone_name: Optional[str] = None,
    ):
        """
        Progressively update person sighting history, multi-camera timeline, and multi-shot appearance.
        """
        now = time.time()
        with self._lock:
            person = self._persons.get(retained_id)
            if not person:
                return

            dt = now - person.last_seen_time
            person.last_seen_time = now
            person.camera_id = camera_id
            if camera_name:
                person.camera_name = camera_name

            # Add sighting to cross-camera history if moved to new camera or significant interval (> 3 sec)
            last_entry = person.camera_history[-1] if person.camera_history else None
            is_new_cam = last_entry is None or last_entry.get("camera_id") != camera_id
            is_time_gap = last_entry and (now - last_entry.get("timestamp", 0) > 3.0)

            if is_new_cam or is_time_gap:
                person.camera_history.append({
                    "timestamp": now,
                    "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                    "camera_id": camera_id,
                    "camera_name": camera_name or f"CAM {camera_id:02d}",
                    "event": "CROSS_CAMERA" if is_new_cam else "TRACKED",
                    "bbox": bbox or [],
                })

            # Multi-observation appearance updating (keep up to 8 best embeddings)
            if embedding is not None and len(embedding) == 512:
                norm = np.linalg.norm(embedding)
                if norm > 1e-6:
                    normed = (embedding / norm).astype(np.float32)
                    if len(person.embeddings) < 8:
                        person.embeddings.append(normed)
                    else:
                        person.embeddings.pop(0)
                        person.embeddings.append(normed)

            # Store additional high-quality snapshot crop (up to 4 snapshots)
            if crop is not None and crop.size > 0 and len(person.snapshots) < 4 and dt > 2.0:
                try:
                    filename = f"{retained_id}_sight_{int(now)}.jpg"
                    abs_path = os.path.join(self.crops_dir, filename)
                    cv2.imwrite(abs_path, crop)
                    person.snapshots.append(f"assets/reid_crops/{filename}")
                except Exception as e:
                    logger.debug(f"Snapshot write error: {e}")

            # Persist state
            self._save_to_db(person)

    # Backward compatibility alias
    def update_suspect(self, persistent_id: str, **kwargs):
        self.update_person(retained_id=persistent_id, **kwargs)

    def _save_to_db(self, person: RetainedPerson):
        """Write updated person record to SQLite."""
        try:
            embs_bytes = b"".join([e.tobytes() for e in person.embeddings])
            hist_json = json.dumps(person.camera_history)
            snaps_json = json.dumps(person.snapshots)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO reid_persons (
                        retained_id, first_seen_time, last_seen_time,
                        camera_id, camera_name, camera_history,
                        embeddings_blob, snapshots_json, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(retained_id) DO UPDATE SET
                        last_seen_time = excluded.last_seen_time,
                        camera_id = excluded.camera_id,
                        camera_name = excluded.camera_name,
                        camera_history = excluded.camera_history,
                        embeddings_blob = excluded.embeddings_blob,
                        snapshots_json = excluded.snapshots_json,
                        status = excluded.status
                """, (
                    person.retained_id,
                    person.first_seen_time,
                    person.last_seen_time,
                    person.camera_id,
                    person.camera_name,
                    hist_json,
                    embs_bytes,
                    snaps_json,
                    person.status,
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"[PersonReIDRegistry] DB write error for {person.retained_id}: {e}")

    def get_person(self, retained_id: str) -> Optional[RetainedPerson]:
        """Retrieve person record by retained ID."""
        with self._lock:
            return self._persons.get(retained_id)

    # Backward compatibility alias
    def get_dossier(self, persistent_id: str) -> Optional[RetainedPerson]:
        return self.get_person(persistent_id)

    def get_all_persons(self) -> List[Dict[str, Any]]:
        """Retrieve all persons as serializable dictionaries for the web dashboard."""
        with self._lock:
            all_list = []
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM reid_persons ORDER BY first_seen_time DESC")
                    for row in cursor.fetchall():
                        pid = row[0]
                        if pid in self._persons:
                            all_list.append(self._persons[pid].to_dict())
                        else:
                            all_list.append({
                                "retained_id": pid,
                                "persistent_id": pid,
                                "first_seen_time": row[1],
                                "first_seen_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row[1])),
                                "last_seen_time": row[2],
                                "last_seen_str": time.strftime("%H:%M:%S", time.localtime(row[2])),
                                "camera_id": row[3],
                                "camera_name": row[4],
                                "camera_history": json.loads(row[5]) if row[5] else [],
                                "snapshots": json.loads(row[7]) if row[7] else [],
                                "dwell_seconds": round(max(0.0, row[2] - row[1]), 1),
                                "status": row[8],
                                "observation_count": len(row[6]) // (512 * 4) if row[6] else 0,
                            })
                return all_list
            except Exception as e:
                logger.error(f"[PersonReIDRegistry] Error fetching all persons: {e}")
                return [p.to_dict() for p in self._persons.values()]

    # Backward compatibility alias
    def get_all_dossiers(self) -> List[Dict[str, Any]]:
        return self.get_all_persons()


# Backward compatibility class alias
SuspectRegistry = PersonReIDRegistry

# Global singleton instance
GLOBAL_REID_REGISTRY = PersonReIDRegistry()
GLOBAL_SUSPECT_REGISTRY = GLOBAL_REID_REGISTRY

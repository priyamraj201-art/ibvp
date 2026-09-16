"""
Entity Database Manager for ByteTrack Surveillance.
Maintains persistent records of all tracked entities (humans, vehicles, objects),
their camera appearances, identification statuses (FRS/ANPR), and sighting audit trails.
"""

import os
import time
import json
import sqlite3
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger


def _ts_to_str(ts: Optional[float]) -> str:
    if ts is None or ts == 0:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


class EntityDatabase:
    """Thread-safe SQLite database for persistent tracked entities and camera sightings."""

    def __init__(self, db_path: str = "tracked_entities.db"):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Main entities registry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    persistent_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    is_identified INTEGER DEFAULT 0,
                    identified_id TEXT DEFAULT 'UNIDENTIFIED',
                    identified_name TEXT DEFAULT 'Unidentified',
                    category TEXT DEFAULT 'NORMAL',
                    confidence REAL DEFAULT 0.0,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    total_sightings INTEGER DEFAULT 1,
                    cameras_seen TEXT DEFAULT '[]',
                    last_camera_id INTEGER DEFAULT 0,
                    last_camera_name TEXT DEFAULT '',
                    thumbnail_path TEXT DEFAULT '',
                    embedding_blob BLOB,
                    notes TEXT DEFAULT '',
                    status TEXT DEFAULT 'ACTIVE'
                );
            """)

            # Detailed camera sightings audit trail
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entity_sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persistent_id TEXT NOT NULL,
                    camera_id INTEGER NOT NULL,
                    camera_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    bbox TEXT DEFAULT '',
                    local_track_id INTEGER DEFAULT 0,
                    is_identified INTEGER DEFAULT 0,
                    identified_id TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    speed REAL DEFAULT 0.0,
                    FOREIGN KEY (persistent_id) REFERENCES entities(persistent_id) ON DELETE CASCADE
                );
            """)

            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_identified ON entities(is_identified);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_identified_id ON entities(identified_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_last_seen ON entities(last_seen);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_pid ON entity_sightings(persistent_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_cam ON entity_sightings(camera_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_ts ON entity_sightings(timestamp);")

            conn.commit()
            conn.close()

    def get_next_id(self, entity_type: str = "HUMAN") -> str:
        """Generate next sequence persistent ID (e.g. PER-0001 or VEH-0001)."""
        prefix = "PER" if entity_type.upper() in ("HUMAN", "PERSON") else ("VEH" if entity_type.upper() in ("VEHICLE", "CAR") else "OBJ")
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT persistent_id FROM entities WHERE persistent_id LIKE ? ORDER BY rowid DESC LIMIT 1", (f"{prefix}-%",))
            row = cursor.fetchone()
            conn.close()

            next_num = 1
            if row and row["persistent_id"]:
                try:
                    curr_num = int(row["persistent_id"].split("-")[1])
                    next_num = curr_num + 1
                except Exception:
                    next_num = 1
            return f"{prefix}-{next_num:04d}"

    def upsert_entity(
        self,
        persistent_id: str,
        entity_type: str,
        first_seen: float,
        last_seen: float,
        cameras_seen: List[str],
        last_camera_id: int,
        last_camera_name: str,
        is_identified: bool = False,
        identified_id: str = "UNIDENTIFIED",
        identified_name: str = "Unidentified",
        category: str = "NORMAL",
        confidence: float = 0.0,
        thumbnail_path: str = "",
        embedding: Optional[np.ndarray] = None,
        notes: str = "",
        status: str = "ACTIVE",
    ) -> bool:
        """Create or update a tracked entity in the database."""
        cams_json = json.dumps(list(set(cameras_seen)))
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None

        with self._lock:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute("SELECT total_sightings, cameras_seen, thumbnail_path, embedding_blob, is_identified, identified_id, identified_name, category, confidence FROM entities WHERE persistent_id = ?", (persistent_id,))
                existing = cursor.fetchone()

                if existing:
                    prev_cams = json.loads(existing["cameras_seen"] or "[]")
                    all_cams = list(set(prev_cams + cameras_seen))
                    new_sightings = (existing["total_sightings"] or 0) + 1

                    # Preserve existing identification if incoming is unidentified
                    final_is_id = existing["is_identified"] or (1 if is_identified else 0)
                    final_id_id = identified_id if is_identified else (existing["identified_id"] or "UNIDENTIFIED")
                    final_name = identified_name if is_identified else (existing["identified_name"] or "Unidentified")
                    final_cat = category if is_identified else (existing["category"] or "NORMAL")
                    final_conf = max(confidence, existing["confidence"] or 0.0) if is_identified else (existing["confidence"] or 0.0)
                    final_thumb = thumbnail_path or existing["thumbnail_path"] or ""
                    final_emb = emb_blob if emb_blob is not None else existing["embedding_blob"]

                    cursor.execute("""
                        UPDATE entities SET
                            is_identified = ?,
                            identified_id = ?,
                            identified_name = ?,
                            category = ?,
                            confidence = ?,
                            last_seen = ?,
                            total_sightings = ?,
                            cameras_seen = ?,
                            last_camera_id = ?,
                            last_camera_name = ?,
                            thumbnail_path = ?,
                            embedding_blob = ?,
                            status = ?
                        WHERE persistent_id = ?
                    """, (
                        final_is_id, final_id_id, final_name, final_cat, final_conf,
                        last_seen, new_sightings, json.dumps(all_cams),
                        last_camera_id, last_camera_name, final_thumb, final_emb,
                        status, persistent_id
                    ))
                else:
                    cursor.execute("""
                        INSERT INTO entities (
                            persistent_id, entity_type, is_identified, identified_id, identified_name,
                            category, confidence, first_seen, last_seen, total_sightings,
                            cameras_seen, last_camera_id, last_camera_name, thumbnail_path,
                            embedding_blob, notes, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        persistent_id, entity_type, 1 if is_identified else 0,
                        identified_id, identified_name, category, confidence,
                        first_seen, last_seen, 1, cams_json,
                        last_camera_id, last_camera_name, thumbnail_path,
                        emb_blob, notes, status
                    ))

                conn.commit()
                conn.close()
                return True
            except Exception as e:
                logger.error(f"[EntityDB] Error upserting entity {persistent_id}: {e}")
                return False

    def record_sighting(
        self,
        persistent_id: str,
        camera_id: int,
        camera_name: str,
        timestamp: float,
        bbox: str = "",
        local_track_id: int = 0,
        is_identified: bool = False,
        identified_id: str = "",
        confidence: float = 0.0,
        speed: float = 0.0,
    ) -> bool:
        """Record an instantaneous sighting observation for an entity."""
        with self._lock:
            try:
                conn = self._get_conn()
                conn.execute("""
                    INSERT INTO entity_sightings (
                        persistent_id, camera_id, camera_name, timestamp,
                        bbox, local_track_id, is_identified, identified_id,
                        confidence, speed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    persistent_id, camera_id, camera_name, timestamp,
                    bbox, local_track_id, 1 if is_identified else 0,
                    identified_id, confidence, speed
                ))
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                logger.error(f"[EntityDB] Error recording sighting for {persistent_id}: {e}")
                return False

    def find_by_identified_id(self, identified_id: str) -> Optional[Dict[str, Any]]:
        """Look up entity by its FRS person_id or ANPR plate_number."""
        if not identified_id or identified_id == "UNIDENTIFIED":
            return None
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM entities WHERE UPPER(identified_id) = ? LIMIT 1",
                (identified_id.strip().upper(),)
            ).fetchone()
            conn.close()
            return dict(row) if row else None

    def search_by_embedding(self, query_emb: np.ndarray, threshold: float = 0.55) -> Optional[Tuple[str, float]]:
        """Search registered entities for a matching face embedding."""
        if query_emb is None or query_emb.size != 512:
            return None
        norm_q = query_emb / (np.linalg.norm(query_emb) + 1e-7)

        with self._lock:
            conn = self._get_conn()
            rows = conn.execute("SELECT persistent_id, embedding_blob FROM entities WHERE embedding_blob IS NOT NULL").fetchall()
            conn.close()

        best_id = None
        best_sim = -1.0

        for r in rows:
            blob = r["embedding_blob"]
            if not blob:
                continue
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.size != 512:
                continue
            norm_e = emb / (np.linalg.norm(emb) + 1e-7)
            sim = float(np.dot(norm_q, norm_e))
            if sim > best_sim:
                best_sim = sim
                best_id = r["persistent_id"]

        if best_sim >= threshold and best_id:
            return best_id, best_sim
        return None

    def get_entity(self, persistent_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single entity record with formatted fields."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM entities WHERE persistent_id = ?", (persistent_id,)).fetchone()
            conn.close()
            if not row:
                return None
            res = dict(row)
            res.pop("embedding_blob", None)
            res["cameras_seen_list"] = json.loads(res.get("cameras_seen") or "[]")
            res["first_seen_str"] = _ts_to_str(res.get("first_seen"))
            res["last_seen_str"] = _ts_to_str(res.get("last_seen"))
            return res

    def get_entities(
        self,
        entity_type: Optional[str] = None,
        is_identified: Optional[bool] = None,
        category: Optional[str] = None,
        camera: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query entities with filters."""
        sql = "SELECT * FROM entities WHERE 1=1"
        params: list = []

        if entity_type and entity_type.upper() != "ALL":
            sql += " AND UPPER(entity_type) = ?"
            params.append(entity_type.upper())

        if is_identified is not None:
            sql += " AND is_identified = ?"
            params.append(1 if is_identified else 0)

        if category and category.upper() != "ALL":
            sql += " AND UPPER(category) = ?"
            params.append(category.upper())

        if camera:
            sql += " AND cameras_seen LIKE ?"
            params.append(f"%{camera}%")

        if query:
            q = f"%{query.strip().upper()}%"
            sql += " AND (UPPER(persistent_id) LIKE ? OR UPPER(identified_id) LIKE ? OR UPPER(identified_name) LIKE ?)"
            params.extend([q, q, q])

        sql += " ORDER BY last_seen DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(sql, params).fetchall()
            conn.close()

        results = []
        for r in rows:
            d = dict(r)
            d.pop("embedding_blob", None)
            d["cameras_seen_list"] = json.loads(d.get("cameras_seen") or "[]")
            d["first_seen_str"] = _ts_to_str(d.get("first_seen"))
            d["last_seen_str"] = _ts_to_str(d.get("last_seen"))
            results.append(d)
        return results

    def get_sightings(self, persistent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get chronological sighting audit logs for an entity."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM entity_sightings WHERE persistent_id = ? ORDER BY timestamp DESC LIMIT ?",
                (persistent_id, limit)
            ).fetchall()
            conn.close()

        results = []
        for r in rows:
            d = dict(r)
            d["timestamp_str"] = _ts_to_str(d.get("timestamp"))
            results.append(d)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate telemetry metrics across all persistent entities."""
        with self._lock:
            conn = self._get_conn()
            total = conn.execute("SELECT COUNT(*) AS c FROM entities").fetchone()["c"]
            humans = conn.execute("SELECT COUNT(*) AS c FROM entities WHERE entity_type = 'HUMAN'").fetchone()["c"]
            vehicles = conn.execute("SELECT COUNT(*) AS c FROM entities WHERE entity_type = 'VEHICLE'").fetchone()["c"]
            identified = conn.execute("SELECT COUNT(*) AS c FROM entities WHERE is_identified = 1").fetchone()["c"]
            unidentified = conn.execute("SELECT COUNT(*) AS c FROM entities WHERE is_identified = 0").fetchone()["c"]
            flagged = conn.execute("SELECT COUNT(*) AS c FROM entities WHERE category IN ('WANTED', 'SUSPECT', 'STOLEN')").fetchone()["c"]
            sightings = conn.execute("SELECT COUNT(*) AS c FROM entity_sightings").fetchone()["c"]
            conn.close()

        return {
            "total_entities": total,
            "total_humans": humans,
            "total_vehicles": vehicles,
            "total_identified": identified,
            "total_unidentified": unidentified,
            "total_flagged": flagged,
            "total_sightings": sightings,
        }

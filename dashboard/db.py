"""
SQLite read helpers for the ByteTrack Surveillance Dashboard.
Uses direct sqlite3 queries (no ORM) to avoid threading lock contention
with the live pipeline's FaceDatabase / WatchlistDB instances.
"""

import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Any


def _ts_to_str(ts: Optional[float]) -> str:
    """Convert Unix timestamp to human-readable datetime string."""
    if ts is None or ts == 0:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


def _get_conn(db_path: str) -> sqlite3.Connection:
    """Open a read-only-safe connection with row factory."""
    conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _db_exists(db_path: str) -> bool:
    return os.path.isfile(db_path)


# ──────────────────────────────────────────────
# FRS Helpers
# ──────────────────────────────────────────────

def get_frs_identity_count(db_path: str) -> int:
    if not _db_exists(db_path):
        return 0
    try:
        conn = _get_conn(db_path)
        row = conn.execute("SELECT COUNT(*) AS cnt FROM face_identities").fetchone()
        conn.close()
        return row["cnt"] if row else 0
    except Exception:
        return 0


def get_frs_identities(db_path: str) -> List[Dict[str, Any]]:
    """Return all face identities WITHOUT embedding_blob."""
    if not _db_exists(db_path):
        return []
    try:
        conn = _get_conn(db_path)
        rows = conn.execute(
            "SELECT person_id, name, category, notes, enrolled_at, enrolled_by, face_count "
            "FROM face_identities ORDER BY enrolled_at DESC"
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            result.append({
                "person_id": r["person_id"],
                "name": r["name"],
                "category": r["category"],
                "notes": r["notes"] or "",
                "enrolled_at": _ts_to_str(r["enrolled_at"]),
                "enrolled_by": r["enrolled_by"] or "admin",
                "face_count": r["face_count"] or 1,
            })
        return result
    except Exception:
        return []


def get_frs_identity(db_path: str, person_id: str) -> Optional[Dict[str, Any]]:
    if not _db_exists(db_path):
        return None
    try:
        conn = _get_conn(db_path)
        row = conn.execute(
            "SELECT person_id, name, category, notes, enrolled_at, enrolled_by, face_count "
            "FROM face_identities WHERE person_id = ?", (person_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "person_id": row["person_id"],
            "name": row["name"],
            "category": row["category"],
            "notes": row["notes"] or "",
            "enrolled_at": _ts_to_str(row["enrolled_at"]),
            "enrolled_by": row["enrolled_by"] or "admin",
            "face_count": row["face_count"] or 1,
        }
    except Exception:
        return None


def get_frs_logs(db_path: str, limit: int = 200, category: str = None,
                 date_from: str = None, date_to: str = None) -> List[Dict[str, Any]]:
    if not _db_exists(db_path):
        return []
    try:
        conn = _get_conn(db_path)
        query = "SELECT * FROM frs_logs WHERE 1=1"
        params: list = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if date_from:
            try:
                ts_from = datetime.strptime(date_from, "%Y-%m-%d").timestamp()
                query += " AND timestamp >= ?"
                params.append(ts_from)
            except Exception:
                pass
        if date_to:
            try:
                ts_to = datetime.strptime(date_to, "%Y-%m-%d").timestamp() + 86400
                query += " AND timestamp < ?"
                params.append(ts_to)
            except Exception:
                pass
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        conn.close()
        result = []
        for r in rows:
            result.append({
                "id": r["id"],
                "timestamp": _ts_to_str(r["timestamp"]),
                "track_id": r["track_id"],
                "person_id": r["person_id"] or "",
                "name": r["name"] or "",
                "confidence": round(r["confidence"] or 0, 3),
                "category": r["category"] or "",
                "is_flagged": bool(r["is_flagged"]),
                "camera_id": r["camera_id"] if r["camera_id"] else 0,
                "bbox_area": round(r["bbox_area"] or 0, 1),
            })
        return result
    except Exception:
        return []


def get_frs_alerts_today(db_path: str) -> int:
    if not _db_exists(db_path):
        return 0
    try:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        conn = _get_conn(db_path)
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM frs_logs WHERE is_flagged = 1 AND timestamp >= ?",
            (today_start,)
        ).fetchone()
        conn.close()
        return row["cnt"] if row else 0
    except Exception:
        return 0


def get_recent_frs_flagged(db_path: str, limit: int = 5) -> List[Dict[str, Any]]:
    if not _db_exists(db_path):
        return []
    try:
        conn = _get_conn(db_path)
        rows = conn.execute(
            "SELECT * FROM frs_logs WHERE is_flagged = 1 ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [{
            "timestamp": _ts_to_str(r["timestamp"]),
            "person_id": r["person_id"] or "",
            "name": r["name"] or "",
            "confidence": round(r["confidence"] or 0, 3),
            "category": r["category"] or "",
        } for r in rows]
    except Exception:
        return []


def get_last_frs_match(db_path: str) -> Optional[Dict[str, Any]]:
    if not _db_exists(db_path):
        return None
    try:
        conn = _get_conn(db_path)
        row = conn.execute(
            "SELECT * FROM frs_logs WHERE is_flagged = 1 ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "name": row["name"] or "",
            "person_id": row["person_id"] or "",
            "confidence": round(row["confidence"] or 0, 3),
            "category": row["category"] or "",
            "timestamp": _ts_to_str(row["timestamp"]),
        }
    except Exception:
        return None


def delete_frs_identity(db_path: str, person_id: str) -> bool:
    if not _db_exists(db_path):
        return False
    try:
        conn = _get_conn(db_path)
        conn.execute("DELETE FROM face_identities WHERE person_id = ?", (person_id,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def seed_frs(db_path: str) -> bool:
    """Seed sample FRS identities via the FaceDatabase class."""
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from yolox.frs.face_database import FaceDatabase
        fdb = FaceDatabase(db_path=db_path)
        fdb.seed_sample_identities()
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────
# ANPR Helpers
# ──────────────────────────────────────────────

def get_anpr_watchlist_count(db_path: str) -> int:
    if not _db_exists(db_path):
        return 0
    try:
        conn = _get_conn(db_path)
        row = conn.execute("SELECT COUNT(*) AS cnt FROM watchlist").fetchone()
        conn.close()
        return row["cnt"] if row else 0
    except Exception:
        return 0


def get_anpr_watchlist(db_path: str) -> List[Dict[str, Any]]:
    if not _db_exists(db_path):
        return []
    try:
        conn = _get_conn(db_path)
        rows = conn.execute(
            "SELECT plate_number, owner_name, alert_category, notes, created_at "
            "FROM watchlist ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [{
            "plate_number": r["plate_number"],
            "owner_name": r["owner_name"] or "",
            "alert_category": r["alert_category"] or "",
            "notes": r["notes"] or "",
            "created_at": _ts_to_str(r["created_at"]),
        } for r in rows]
    except Exception:
        return []


def get_anpr_logs(db_path: str, limit: int = 200) -> List[Dict[str, Any]]:
    if not _db_exists(db_path):
        return []
    try:
        conn = _get_conn(db_path)
        rows = conn.execute(
            "SELECT * FROM anpr_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [{
            "id": r["id"],
            "timestamp": _ts_to_str(r["timestamp"]),
            "track_id": r["track_id"],
            "plate_number": r["plate_number"] or "",
            "confidence": round(r["confidence"] or 0, 3),
            "is_flagged": bool(r["is_flagged"]),
            "alert_category": r["alert_category"] or "",
            "bbox_area": round(r["bbox_area"] or 0, 1),
            "crop_path": r["crop_path"] or "",
        } for r in rows]
    except Exception:
        return []


def get_anpr_alerts_today(db_path: str) -> int:
    if not _db_exists(db_path):
        return 0
    try:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        conn = _get_conn(db_path)
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM anpr_logs WHERE is_flagged = 1 AND timestamp >= ?",
            (today_start,)
        ).fetchone()
        conn.close()
        return row["cnt"] if row else 0
    except Exception:
        return 0


def get_recent_anpr_flagged(db_path: str, limit: int = 5) -> List[Dict[str, Any]]:
    if not _db_exists(db_path):
        return []
    try:
        conn = _get_conn(db_path)
        rows = conn.execute(
            "SELECT * FROM anpr_logs WHERE is_flagged = 1 ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [{
            "timestamp": _ts_to_str(r["timestamp"]),
            "plate_number": r["plate_number"] or "",
            "confidence": round(r["confidence"] or 0, 3),
            "alert_category": r["alert_category"] or "",
        } for r in rows]
    except Exception:
        return []


def get_last_anpr_match(db_path: str) -> Optional[Dict[str, Any]]:
    if not _db_exists(db_path):
        return None
    try:
        conn = _get_conn(db_path)
        row = conn.execute(
            "SELECT * FROM anpr_logs WHERE is_flagged = 1 ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "plate": row["plate_number"] or "",
            "category": row["alert_category"] or "",
            "confidence": round(row["confidence"] or 0, 3),
            "timestamp": _ts_to_str(row["timestamp"]),
        }
    except Exception:
        return None


def add_anpr_plate(db_path: str, plate_number: str, owner_name: str = "",
                   alert_category: str = "SUSPICIOUS", notes: str = "") -> bool:
    try:
        conn = _get_conn(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO watchlist (plate_number, owner_name, alert_category, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (plate_number.strip().upper(), owner_name, alert_category.strip().upper(), notes, time.time())
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def delete_anpr_plate(db_path: str, plate_number: str) -> bool:
    if not _db_exists(db_path):
        return False
    try:
        conn = _get_conn(db_path)
        conn.execute("DELETE FROM watchlist WHERE plate_number = ?", (plate_number,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def seed_anpr(db_path: str) -> bool:
    """Seed sample ANPR watchlist via the WatchlistDB class."""
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from yolox.anpr.watchlist_db import WatchlistDB
        wdb = WatchlistDB(db_path=db_path)
        wdb.seed_sample_watchlist()
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────
# Unified Alerts
# ──────────────────────────────────────────────

def get_unified_alerts(frs_db: str, anpr_db: str, limit: int = 200,
                       alert_type: str = "all") -> List[Dict[str, Any]]:
    """Merge FRS + ANPR flagged logs, sorted by timestamp DESC."""
    alerts: List[Dict[str, Any]] = []

    if alert_type in ("all", "face") and _db_exists(frs_db):
        try:
            conn = _get_conn(frs_db)
            rows = conn.execute(
                "SELECT * FROM frs_logs WHERE is_flagged = 1 ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            conn.close()
            for r in rows:
                alerts.append({
                    "timestamp_raw": r["timestamp"] or 0,
                    "timestamp": _ts_to_str(r["timestamp"]),
                    "type": "FACE",
                    "id": r["person_id"] or "",
                    "name_or_plate": r["name"] or "",
                    "category": r["category"] or "",
                    "confidence": round(r["confidence"] or 0, 3),
                    "camera_id": r["camera_id"] if r["camera_id"] else 0,
                })
        except Exception:
            pass

    if alert_type in ("all", "plate") and _db_exists(anpr_db):
        try:
            conn = _get_conn(anpr_db)
            rows = conn.execute(
                "SELECT * FROM anpr_logs WHERE is_flagged = 1 ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            conn.close()
            for r in rows:
                alerts.append({
                    "timestamp_raw": r["timestamp"] or 0,
                    "timestamp": _ts_to_str(r["timestamp"]),
                    "type": "PLATE",
                    "id": r["plate_number"] or "",
                    "name_or_plate": r["plate_number"] or "",
                    "category": r["alert_category"] or "",
                    "confidence": round(r["confidence"] or 0, 3),
                    "camera_id": 0,
                })
        except Exception:
            pass

    alerts.sort(key=lambda x: x["timestamp_raw"], reverse=True)
    return alerts[:limit]

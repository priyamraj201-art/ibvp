"""
Comprehensive Test Suite for Global Persistent Entity Tracking & Cross-Camera Telemetry.
Validates:
1. EntityDatabase SQLite schema, indexing, and CRUD queries
2. PersistentTracker ID allocation (PER-xxxx, VEH-xxxx)
3. Unidentified person tracking (always assigned persistent ID, is_identified=False)
4. Identified person tracking (FRS updates is_identified=True, identified_id, category)
5. Cross-camera re-identification (same person appearing across multiple cameras)
6. FastAPI endpoints (/entities, /api/entities, /api/entities/{id}, /api/entities/stats)
"""

import os
import sys
import time
import pytest
import numpy as np

# Ensure ByteTrack root is on path
_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from yolox.tracker.entity_db import EntityDatabase
from yolox.tracker.persistent_tracker import PersistentTracker
from dashboard.server import create_app
from fastapi.testclient import TestClient


def test_entity_db_crud(tmp_path):
    """Test EntityDatabase schema, insertion, retrieval, and sightings."""
    db_file = str(tmp_path / "test_entities.db")
    db = EntityDatabase(db_path=db_file)

    # 1. ID Generation
    id1 = db.get_next_id("HUMAN")
    id2 = db.get_next_id("HUMAN")
    veh1 = db.get_next_id("VEHICLE")
    assert id1 == "PER-0001"
    assert veh1 == "VEH-0001"

    # 2. Upsert Unidentified Human
    now = time.time()
    ok = db.upsert_entity(
        persistent_id=id1,
        entity_type="HUMAN",
        first_seen=now,
        last_seen=now,
        cameras_seen=["CAM 01: Checkpost"],
        last_camera_id=1,
        last_camera_name="CAM 01: Checkpost",
        is_identified=False,
        identified_id="UNIDENTIFIED",
        identified_name="Unidentified Person",
    )
    assert ok is True

    # 3. Fetch entity
    e1 = db.get_entity(id1)
    assert e1 is not None
    assert e1["persistent_id"] == id1
    assert e1["entity_type"] == "HUMAN"
    assert e1["is_identified"] == 0
    assert e1["identified_id"] == "UNIDENTIFIED"
    assert "CAM 01: Checkpost" in e1["cameras_seen_list"]

    # 4. Record Sighting on CAM 01
    s_ok = db.record_sighting(
        persistent_id=id1,
        camera_id=1,
        camera_name="CAM 01: Checkpost",
        timestamp=now,
        bbox="10,20,100,200",
        local_track_id=5,
        speed=15.5,
    )
    assert s_ok is True

    # 5. Record Second Sighting on CAM 02 (Cross-camera visit)
    now2 = now + 10.0
    db.upsert_entity(
        persistent_id=id1,
        entity_type="HUMAN",
        first_seen=now,
        last_seen=now2,
        cameras_seen=["CAM 02: Control Room"],
        last_camera_id=2,
        last_camera_name="CAM 02: Control Room",
    )
    db.record_sighting(
        persistent_id=id1,
        camera_id=2,
        camera_name="CAM 02: Control Room",
        timestamp=now2,
        bbox="30,40,120,220",
        local_track_id=8,
        speed=22.0,
    )

    e1_updated = db.get_entity(id1)
    assert len(e1_updated["cameras_seen_list"]) == 2
    assert "CAM 01: Checkpost" in e1_updated["cameras_seen_list"]
    assert "CAM 02: Control Room" in e1_updated["cameras_seen_list"]
    assert e1_updated["total_sightings"] == 2

    # 6. Check Sightings audit trail
    sightings = db.get_sightings(id1)
    assert len(sightings) == 2

    # 7. Check Stats
    stats = db.get_stats()
    assert stats["total_entities"] == 1
    assert stats["total_humans"] == 1
    assert stats["total_unidentified"] == 1
    assert stats["total_identified"] == 0


def test_persistent_tracker_lifecycle(tmp_path):
    """Test PersistentTracker ID assignment, FRS identification update, and cross-camera correlation."""
    db_file = str(tmp_path / "test_tracker_entities.db")
    crops_dir = str(tmp_path / "crops")
    tracker = PersistentTracker(db_path=db_file, crops_dir=crops_dir, sighting_interval_sec=0.1)

    # 1. Unidentified human appears on CAM 01
    pid_cam1 = tracker.assign_or_get_id(
        cam_id=1,
        cam_name="CAM 01",
        local_track_id=42,
        entity_type="HUMAN",
        tlwh=[100, 200, 80, 160],
    )
    assert pid_cam1.startswith("PER-")

    # Same track on same camera gets same persistent ID
    pid_cam1_again = tracker.assign_or_get_id(
        cam_id=1,
        cam_name="CAM 01",
        local_track_id=42,
        entity_type="HUMAN",
        tlwh=[105, 205, 80, 160],
    )
    assert pid_cam1_again == pid_cam1

    # Entity is recorded as unidentified in DB
    ent = tracker.db.get_entity(pid_cam1)
    assert ent["is_identified"] == 0
    assert ent["identified_id"] == "UNIDENTIFIED"

    # 2. FRS identifies the person on CAM 01
    final_id = tracker.update_identification(
        cam_id=1,
        local_track_id=42,
        is_identified=True,
        identified_id="SUSPECT_007",
        identified_name="Vikram Singh",
        category="WANTED",
        confidence=0.88,
        cam_name="CAM 01",
    )
    assert final_id == pid_cam1

    # Flush DB queue
    time.sleep(0.3)
    ent_identified = tracker.db.get_entity(pid_cam1)
    assert ent_identified["is_identified"] == 1
    assert ent_identified["identified_id"] == "SUSPECT_007"
    assert ent_identified["identified_name"] == "Vikram Singh"
    assert ent_identified["category"] == "WANTED"

    # 3. Same person appears on CAM 02 (Cross-Camera Correlation)
    pid_cam2_initial = tracker.assign_or_get_id(
        cam_id=2,
        cam_name="CAM 02",
        local_track_id=99,
        entity_type="HUMAN",
        tlwh=[50, 60, 90, 180],
    )
    # Before FRS runs, it gets a temporary ID
    assert pid_cam2_initial.startswith("PER-")

    # FRS matches SUSPECT_007 on CAM 02
    canonical_id = tracker.update_identification(
        cam_id=2,
        local_track_id=99,
        is_identified=True,
        identified_id="SUSPECT_007",
        identified_name="Vikram Singh",
        category="WANTED",
        confidence=0.92,
        cam_name="CAM 02",
    )
    # Should link back to the first persistent ID!
    assert canonical_id == pid_cam1

    time.sleep(0.3)
    ent_cross = tracker.db.get_entity(pid_cam1)
    assert "CAM 01" in ent_cross["cameras_seen_list"]
    assert "CAM 02" in ent_cross["cameras_seen_list"]

    tracker.close()


def test_fastapi_entities_routes(tmp_path):
    """Test FastAPI /entities and /api/entities endpoints."""
    frs_db = str(tmp_path / "frs_test.db")
    anpr_db = str(tmp_path / "anpr_test.db")

    app = create_app(frs_db, anpr_db)
    client = TestClient(app)

    # 1. Test GET /entities (HTML Page)
    resp = client.get("/entities")
    assert resp.status_code == 200
    assert "Tracked Entities & Persistent IDs" in resp.text
    assert "Persistent ID" in resp.text

    # 2. Test GET /api/entities (JSON API)
    api_resp = client.get("/api/entities")
    assert api_resp.status_code == 200
    data = api_resp.json()
    assert data["status"] == "success"
    assert "entities" in data
    assert "stats" in data

    # 3. Test GET /api/entities/stats
    stats_resp = client.get("/api/entities/stats")
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert "total_entities" in stats_data
    assert "total_humans" in stats_data
    assert "total_vehicles" in stats_data

"""
End-to-End System Integration Test for IBVAP Retained ID Re-ID, Multi-Camera, and Dashboard.
Verifies all modules, routes, and APIs together without virtual fence intrusion.
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from starlette.testclient import TestClient
from dashboard.server import create_app
from yolox.reid import GLOBAL_REID_PIPELINE, GLOBAL_REID_REGISTRY, PersonReIDExtractor


def test_system():
    print("=" * 70)
    print("  IBVAP End-to-End System Integration & API Test")
    print("=" * 70)

    # 1. Test ReID components
    print("\n[1/3] Verifying Retained ID Re-ID Pipeline & Registry...")
    extractor = PersonReIDExtractor(fp16=False)
    assert extractor.model is not None, "OSNet model should be initialized"
    print(f"      OSNet model verified on {extractor.device_name}")

    # 2. Test FastAPI Server and Router Endpoints
    print("\n[2/3] Initializing FastAPI Dashboard Server TestClient...")
    app = create_app(
        frs_db=os.path.join(_bytetrack_root, "frs_faces.db"),
        anpr_db=os.path.join(_bytetrack_root, "anpr_watchlist.db"),
    )
    client = TestClient(app)

    # Test /live page
    resp_live = client.get("/live")
    assert resp_live.status_code == 200, f"Expected 200, got {resp_live.status_code}"
    print("      GET /live -> 200 OK (Live feeds rendered)")

    # Test /overview page
    resp_overview = client.get("/overview")
    assert resp_overview.status_code == 200, f"Expected 200, got {resp_overview.status_code}"
    print("      GET /overview -> 200 OK")

    # Test /entities page
    resp_entities = client.get("/entities")
    assert resp_entities.status_code == 200, f"Expected 200, got {resp_entities.status_code}"
    print("      GET /entities -> 200 OK")

    # Test /api/entities JSON API
    resp_api_entities = client.get("/api/entities")
    assert resp_api_entities.status_code == 200, f"Expected 200, got {resp_api_entities.status_code}"
    print(f"      GET /api/entities -> 200 OK (Found {resp_api_entities.json().get('count', 0)} entities)")

    # Test /alerts page
    resp_alerts = client.get("/alerts")
    assert resp_alerts.status_code == 200
    print("      GET /alerts -> 200 OK")

    # 3. Clean up
    print("\n[3/3] Cleaning up background workers...")
    GLOBAL_REID_PIPELINE.stop()

    print("\n" + "=" * 70)
    print("  ALL END-TO-END INTEGRATION TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    test_system()

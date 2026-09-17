"""
Verifies the /live dashboard page includes the phone-camera pairing UI.
"""

import os
import sys
import pytest

_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from dashboard.server import create_app
from fastapi.testclient import TestClient


def test_live_page_includes_add_phone_camera_button():
    frs_db = os.path.join(_bytetrack_root, "frs_faces.db")
    anpr_db = os.path.join(_bytetrack_root, "anpr_watchlist.db")
    client = TestClient(create_app(frs_db, anpr_db))

    res = client.get("/live")
    assert res.status_code == 200
    assert "Add Phone Camera" in res.text
    assert 'id="phonecam-modal"' in res.text
    assert "openPhoneCamModal" in res.text


def test_live_page_stats_polling_rerenders_grid_when_camera_set_changes():
    """Regression test: a camera registered after the /live page has already
    loaded (e.g. a phone pairing mid-session) must get a tile. The 1s stats
    poller used to only patch existing tiles by id via updateTileBadges(),
    so a newly-added camera with no existing DOM tile never appeared until
    the page was reloaded. The poller must detect a change in the camera
    set and call renderGridView() again."""
    frs_db = os.path.join(_bytetrack_root, "frs_faces.db")
    anpr_db = os.path.join(_bytetrack_root, "anpr_watchlist.db")
    client = TestClient(create_app(frs_db, anpr_db))

    res = client.get("/live")
    assert res.status_code == 200
    text = res.text

    poll_fn = text[text.index("function startStatsPolling"):text.index("function getClusterPipelineConfig")]
    assert "cameraSetChanged" in poll_fn
    assert "renderGridView()" in poll_fn


if __name__ == "__main__":
    pytest.main(["-v", __file__])

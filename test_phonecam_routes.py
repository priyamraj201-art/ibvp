"""
Integration tests for the phone camera pairing HTTP routes.
"""

import os
import sys
import base64
import pytest

_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from dashboard.server import create_app
from fastapi.testclient import TestClient


def _client():
    frs_db = os.path.join(_bytetrack_root, "frs_faces.db")
    anpr_db = os.path.join(_bytetrack_root, "anpr_watchlist.db")
    return TestClient(create_app(frs_db, anpr_db))


def test_create_pair_token_returns_join_url_and_qr():
    client = _client()
    res = client.post("/api/phonecam/pair")
    assert res.status_code == 200
    data = res.json()
    assert "join_url" in data and "/phonecam/join/" in data["join_url"]
    assert "qr_png_base64" in data
    raw = base64.b64decode(data["qr_png_base64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert data["expires_in"] == 600


def test_join_page_renders_for_valid_token():
    client = _client()
    token = client.post("/api/phonecam/pair").json()["join_url"].rsplit("/", 1)[-1]
    res = client.get(f"/phonecam/join/{token}")
    assert res.status_code == 200
    assert "Start Sharing" in res.text


def test_join_page_shows_expired_for_unknown_token():
    client = _client()
    res = client.get("/phonecam/join/not-a-real-token")
    assert res.status_code == 404
    assert "expired" in res.text.lower()


if __name__ == "__main__":
    pytest.main(["-v", __file__])

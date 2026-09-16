"""
ByteTrack Surveillance Dashboard — FastAPI Server Entrypoint.

Usage:
    python dashboard/server.py [--host 0.0.0.0] [--port 8000] [--frs-db PATH] [--anpr-db PATH]
"""

import os
import sys
import argparse

# Ensure ByteTrack root is on sys.path for all imports
_BYTETRACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DASHBOARD_ROOT = os.path.dirname(os.path.abspath(__file__))
if _BYTETRACK_ROOT not in sys.path:
    sys.path.insert(0, _BYTETRACK_ROOT)
if _DASHBOARD_ROOT not in sys.path:
    sys.path.insert(0, _DASHBOARD_ROOT)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import routers
from dashboard.routers import live, overview, frs, anpr, alerts, settings, entities


def create_app(frs_db: str, anpr_db: str) -> FastAPI:
    app = FastAPI(
        title="IBVAP Surveillance Dashboard",
        description="ByteTrack Multi-Modal Surveillance Dashboard",
        version="1.0.0",
    )

    # Mount assets for entity thumbnail crops
    assets_dir = os.path.join(_BYTETRACK_ROOT, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    # Store DB paths in app state for access by routers
    app.state.frs_db = frs_db
    app.state.anpr_db = anpr_db

    # Include routers
    app.include_router(live.router)
    app.include_router(overview.router)
    app.include_router(entities.router)
    app.include_router(frs.router)
    app.include_router(anpr.router)
    app.include_router(alerts.router)
    app.include_router(settings.router)

    @app.on_event("shutdown")
    def shutdown_event():
        from dashboard.stream_server import MULTI_CAMERA_MANAGER
        MULTI_CAMERA_MANAGER.stop_all()
        from yolox.reid import GLOBAL_REID_PIPELINE
        GLOBAL_REID_PIPELINE.stop()

    return app


def main():
    parser = argparse.ArgumentParser(description="ByteTrack Surveillance Dashboard")
    parser.add_argument("--frs-db", default=None, help="Path to FRS SQLite database (default: ../frs_faces.db)")
    parser.add_argument("--anpr-db", default=None, help="Path to ANPR SQLite database (default: ../anpr_watchlist.db)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()

    # Resolve DB paths relative to ByteTrack root
    frs_db = args.frs_db or os.path.join(_BYTETRACK_ROOT, "frs_faces.db")
    anpr_db = args.anpr_db or os.path.join(_BYTETRACK_ROOT, "anpr_watchlist.db")
    frs_db = os.path.abspath(frs_db)
    anpr_db = os.path.abspath(anpr_db)

    print("=" * 60)
    print("  IBVAP Surveillance Dashboard")
    print("=" * 60)
    print(f"  FRS Database : {frs_db}")
    print(f"  ANPR Database: {anpr_db}")
    print(f"  FRS DB exists: {os.path.isfile(frs_db)}")
    print(f"  ANPR DB exists: {os.path.isfile(anpr_db)}")
    print(f"  ByteTrack Root: {_BYTETRACK_ROOT}")
    print("=" * 60)
    print(f"  Dashboard running at http://localhost:{args.port}")
    print("=" * 60)

    app = create_app(frs_db, anpr_db)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

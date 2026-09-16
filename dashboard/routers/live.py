"""
Live Feed Router — Multi-Camera MJPEG Streaming, Cluster Control, and Registry APIs.
"""

import os
import sys
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# Ensure ByteTrack root is on path
_bytetrack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from dashboard.camera_manager import CAMERA_REGISTRY
from dashboard.stream_server import MULTI_CAMERA_MANAGER, STREAM_STATE, mjpeg_generator

router = APIRouter()

_templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_templates_dir)


# ──────────────────────────────────────────────
# Page Views
# ──────────────────────────────────────────────

@router.get("/live")
async def live_page(request: Request):
    """Render the interactive multi-camera grid surveillance command center."""
    cameras = CAMERA_REGISTRY.load_cameras()
    return templates.TemplateResponse(
        request=request,
        name="live.html",
        context={"request": request, "initial_cameras": cameras},
    )


# ──────────────────────────────────────────────
# MJPEG Video Streaming Endpoints
# ──────────────────────────────────────────────

@router.get("/stream/video/{cam_id}")
async def stream_camera_video(cam_id: int):
    """Stream low-latency MJPEG multipart video feed for a specific camera ID."""
    state = MULTI_CAMERA_MANAGER.get_state(cam_id)
    return StreamingResponse(
        mjpeg_generator(state),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/stream/video")
async def stream_primary_video():
    """Backward-compatible endpoint streaming Camera 0 / Primary camera feed."""
    return StreamingResponse(
        mjpeg_generator(STREAM_STATE),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ──────────────────────────────────────────────
# Multi-Camera Cluster Control APIs
# ──────────────────────────────────────────────

@router.get("/api/cameras")
async def get_cameras():
    """Return list of all configured cameras with their current live status and metrics."""
    return JSONResponse(MULTI_CAMERA_MANAGER.get_all_stats())


@router.post("/api/cameras")
async def save_camera(request: Request):
    """Add a new camera or update an existing camera configuration."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON body"}, status_code=400)

    saved_cam = CAMERA_REGISTRY.upsert_camera(body)
    cid = saved_cam.get("id", 0)

    # If camera is currently active, re-sync its state
    state = MULTI_CAMERA_MANAGER.get_state(cid)
    state.update_stats(name=saved_cam.get("name"), location=saved_cam.get("location"))

    return JSONResponse({"status": "success", "camera": saved_cam})


@router.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: int):
    """Delete a camera from the registry and stop its stream if running."""
    MULTI_CAMERA_MANAGER.stop_camera(cam_id)
    success = CAMERA_REGISTRY.delete_camera(cam_id)
    return JSONResponse({"status": "deleted" if success else "not_found", "cam_id": cam_id})


@router.post("/api/cameras/{cam_id}/toggle")
async def toggle_camera(cam_id: int, request: Request):
    """Toggle enabled/disabled state of a camera."""
    try:
        body = await request.json()
        enabled = body.get("enabled", None)
    except Exception:
        enabled = None

    cam = CAMERA_REGISTRY.toggle_camera(cam_id, enabled)
    if not cam:
        return JSONResponse({"status": "error", "message": f"Camera {cam_id} not found"}, status_code=404)

    if not cam.get("enabled"):
        MULTI_CAMERA_MANAGER.stop_camera(cam_id)

    return JSONResponse({"status": "success", "camera": cam})


@router.post("/api/cameras/{cam_id}/start")
async def start_camera_stream(cam_id: int, request: Request):
    """Start ingestion and tracking pipeline for a specific camera."""
    try:
        config = await request.json()
    except Exception:
        config = {}

    success = MULTI_CAMERA_MANAGER.start_camera(cam_id, config)
    state = MULTI_CAMERA_MANAGER.get_state(cam_id)

    return JSONResponse({
        "status": "started" if success else "failed",
        "cam_id": cam_id,
        "camera_stats": state.get_stats(),
    })


@router.post("/api/cameras/{cam_id}/stop")
async def stop_camera_stream(cam_id: int):
    """Stop ingestion and tracking pipeline for a specific camera."""
    success = MULTI_CAMERA_MANAGER.stop_camera(cam_id)
    return JSONResponse({
        "status": "stopped" if success else "not_running",
        "cam_id": cam_id,
    })


@router.post("/api/cameras/start_all")
async def start_all_cameras(request: Request):
    """Concurrently start all enabled cameras in the cluster."""
    try:
        config = await request.json()
    except Exception:
        config = {}

    results = MULTI_CAMERA_MANAGER.start_all(config)
    return JSONResponse({
        "status": "started_all",
        "results": results,
    })


@router.post("/api/cameras/stop_all")
async def stop_all_cameras():
    """Concurrently stop all active cameras in the cluster."""
    results = MULTI_CAMERA_MANAGER.stop_all()
    return JSONResponse({
        "status": "stopped_all",
        "results": results,
    })


@router.get("/api/cameras/{cam_id}/stats")
async def get_camera_stats(cam_id: int):
    """Get real-time telemetry stats for a specific camera."""
    state = MULTI_CAMERA_MANAGER.get_state(cam_id)
    return JSONResponse(state.get_stats())


@router.get("/api/cameras/stats_all")
async def get_all_camera_stats():
    """Get real-time cluster telemetry stats across all cameras."""
    return JSONResponse(MULTI_CAMERA_MANAGER.get_all_stats())


# ──────────────────────────────────────────────
# Backward Compatibility Endpoints
# ──────────────────────────────────────────────

@router.post("/api/stream/start")
async def start_stream(request: Request):
    """Legacy start API: starts Camera 0 or specified source."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    source = body.get("source", "0")
    # Update camera 0 url if provided
    cam = CAMERA_REGISTRY.get_camera(0)
    if cam:
        cam["url"] = source
        CAMERA_REGISTRY.upsert_camera(cam)

    MULTI_CAMERA_MANAGER.start_camera(0, body)
    return JSONResponse({
        "status": "started",
        "source": source,
        "device": body.get("device", "gpu"),
        "fp16": body.get("fp16", True),
    })


@router.post("/api/stream/stop")
async def stop_stream():
    """Legacy stop API: stops Camera 0."""
    MULTI_CAMERA_MANAGER.stop_camera(0)
    return JSONResponse({"status": "stopped"})


@router.get("/api/live/stats")
async def live_stats():
    """Legacy stats API: returns Camera 0 telemetry."""
    return JSONResponse(STREAM_STATE.get_stats())


# ──────────────────────────────────────────────
# Phone Camera "Shared Perception" — Pairing & Join
# ──────────────────────────────────────────────

@router.post("/api/phonecam/pair")
async def create_phonecam_pair(request: Request):
    """Generate a short-lived pairing token + QR code for a phone to join as a camera."""
    from dashboard.phonecam import PHONE_DEVICE_REGISTRY, get_lan_ip, generate_qr_png_base64

    token = PHONE_DEVICE_REGISTRY.create_pair_token()
    lan_ip = get_lan_ip()
    port = request.url.port or 8000
    join_url = f"http://{lan_ip}:{port}/phonecam/join/{token}"
    qr_b64 = generate_qr_png_base64(join_url)

    return JSONResponse({
        "join_url": join_url,
        "qr_png_base64": qr_b64,
        "expires_in": PHONE_DEVICE_REGISTRY.PAIR_TOKEN_TTL_SECONDS,
    })


@router.get("/phonecam/join/{pair_token}")
async def phonecam_join_page(request: Request, pair_token: str):
    """Mobile-facing page a phone opens after scanning the pairing QR code."""
    from dashboard.phonecam import PHONE_DEVICE_REGISTRY

    valid = PHONE_DEVICE_REGISTRY.is_token_valid(pair_token)
    response = templates.TemplateResponse(
        request=request,
        name="phonecam_join.html",
        context={"request": request, "pair_token": pair_token, "expired": not valid},
        status_code=200 if valid else 404,
    )
    return response

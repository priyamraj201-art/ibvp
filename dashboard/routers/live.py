"""
Live feed router — MJPEG streaming + stream control APIs.
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

from dashboard.stream_server import STREAM_STATE, PipelineThread, mjpeg_generator

router = APIRouter()

_templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_templates_dir)

# Module-level pipeline thread reference
_pipeline_thread: Optional[PipelineThread] = None


@router.get("/live")
async def live_page(request: Request):
    return templates.TemplateResponse(request=request, name="live.html", context={"request": request})


@router.get("/stream/video")
async def stream_video():
    return StreamingResponse(
        mjpeg_generator(STREAM_STATE),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/api/stream/start")
async def start_stream(request: Request):
    global _pipeline_thread

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Stop existing thread if running
    if _pipeline_thread is not None and _pipeline_thread.is_alive():
        _pipeline_thread.stop()
        _pipeline_thread.join(timeout=3)
        _pipeline_thread = None

    STREAM_STATE.reset()

    config = {
        "source": body.get("source", "1"),
        "device": body.get("device", "gpu"),
        "fp16": bool(body.get("fp16", True)),
        "enable_frs": bool(body.get("enable_frs", False)),
        "enable_anpr": bool(body.get("enable_anpr", False)),
        "enable_motion_alert": bool(body.get("enable_motion_alert", True)),
        "detect_skip": int(body.get("detect_skip", 1)),
        "exp_file": body.get("exp_file", "exps/example/mot/yolox_x_mix_det.py"),
        "ckpt_file": body.get("ckpt_file", "pretrained/bytetrack_x_mot17.pth.tar"),
        "frs_db": body.get("frs_db", "frs_faces.db"),
        "anpr_db": body.get("anpr_db", "anpr_watchlist.db"),
        "frs_min_area": float(body.get("frs_min_area", 1500.0)),
        "frs_threshold": float(body.get("frs_threshold", 0.60)),
        "frs_workers": int(body.get("frs_workers", 1)),
        "anpr_min_area": float(body.get("anpr_min_area", 800.0)),
        "anpr_workers": int(body.get("anpr_workers", 1)),
        "track_thresh": float(body.get("track_thresh", 0.25)),
        "track_buffer": int(body.get("track_buffer", 30)),
        "match_thresh": float(body.get("match_thresh", 0.8)),
        "detector_mode": body.get("detector_mode", "single_class_test"),
        "alert_low": float(body.get("alert_low", 20.0)),
        "alert_med": float(body.get("alert_med", 60.0)),
        "alert_high": float(body.get("alert_high", 120.0)),
    }

    _pipeline_thread = PipelineThread(config, STREAM_STATE)
    _pipeline_thread.start()

    return JSONResponse({
        "status": "started",
        "source": config["source"],
        "device": config["device"],
        "fp16": config["fp16"],
    })


@router.post("/api/stream/stop")
async def stop_stream():
    global _pipeline_thread

    if _pipeline_thread is not None and _pipeline_thread.is_alive():
        _pipeline_thread.stop()
        _pipeline_thread.join(timeout=3)
        _pipeline_thread = None
        STREAM_STATE.update_stats(is_running=False)
        return JSONResponse({"status": "stopped"})

    return JSONResponse({"status": "not_running"})


@router.get("/api/live/stats")
async def live_stats():
    return JSONResponse(STREAM_STATE.get_stats())

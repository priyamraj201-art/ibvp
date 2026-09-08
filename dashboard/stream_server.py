"""
Live streaming server for the ByteTrack Surveillance Dashboard.
Contains: LiveStreamState (thread-safe shared state), PipelineThread (background worker),
and mjpeg_generator (async MJPEG frame generator).
"""

import os
import sys
import time
import threading
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# Windows OpenMP runtime collision fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure ByteTrack root and tools are on path
_bytetrack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_tools_dir = os.path.join(_bytetrack_root, "tools")
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

import cv2
import numpy as np
import torch

from loguru import logger
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.alert_system import MotionAlertSystem
from yolox.anpr import ANPRPipeline, ANPRVisualizer
from yolox.frs import FRSPipeline, FRSVisualizer
from yolox.routing import TrackRouter, DetectorMode
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer
from demo_track import Predictor


# ──────────────────────────────────────────────
# Shared State
# ──────────────────────────────────────────────

@dataclass
class LiveStats:
    is_running: bool = False
    source: str = ""
    device: str = "GPU"
    fps: float = 0.0
    frame_id: int = 0
    active_tracks: int = 0
    human_count: int = 0
    vehicle_count: int = 0
    motion_alert_level: str = "NORMAL"
    motion_alert_color: str = "#22c55e"
    error_message: str = ""
    last_frs_match: Optional[Dict] = None
    last_anpr_match: Optional[Dict] = None
    frs_alerts_session: int = 0
    anpr_alerts_session: int = 0


class LiveStreamState:
    def __init__(self):
        self._lock = threading.Lock()
        self.stats = LiveStats()
        self._frame: Optional[bytes] = None  # latest JPEG bytes

    def set_frame(self, jpeg_bytes: bytes):
        with self._lock:
            self._frame = jpeg_bytes

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def update_stats(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.stats, k):
                    setattr(self.stats, k, v)

    def get_stats(self) -> dict:
        with self._lock:
            s = self.stats
            return {
                "is_running": s.is_running,
                "source": s.source,
                "device": s.device,
                "fps": round(s.fps, 1),
                "frame_id": s.frame_id,
                "active_tracks": s.active_tracks,
                "human_count": s.human_count,
                "vehicle_count": s.vehicle_count,
                "motion_alert_level": s.motion_alert_level,
                "motion_alert_color": s.motion_alert_color,
                "error_message": s.error_message,
                "frs_alerts_session": s.frs_alerts_session,
                "anpr_alerts_session": s.anpr_alerts_session,
                "last_frs_match": s.last_frs_match,
                "last_anpr_match": s.last_anpr_match,
            }

    def reset(self):
        with self._lock:
            self.stats = LiveStats()
            self._frame = None


# Global singleton
STREAM_STATE = LiveStreamState()


# ──────────────────────────────────────────────
# Camera / Video Capture Opener Helper
# ──────────────────────────────────────────────

def open_video_capture(source_str: str) -> tuple[Optional[cv2.VideoCapture], str, bool]:
    """
    Robust camera and video opener supporting DirectShow on Windows.
    Returns (cap, resolved_source_label, is_video_file).
    """
    source_str = str(source_str).strip()

    # Numeric camera index (e.g. "0", "1", "2")
    if source_str.isdigit():
        cam_id = int(source_str)
        # 1. Try DirectShow
        try:
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap, f"Camera {cam_id} (DSHOW)", False
        except Exception:
            pass

        # 2. Try default backend
        try:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap, f"Camera {cam_id}", False
        except Exception:
            pass

        return None, f"Camera index {cam_id} could not be opened", False

    # Video file or stream URL
    resolved_path = source_str
    if not os.path.isabs(source_str):
        candidate = os.path.join(_bytetrack_root, source_str)
        if os.path.isfile(candidate):
            resolved_path = candidate

    cap = cv2.VideoCapture(resolved_path)
    if cap.isOpened():
        is_file = os.path.isfile(resolved_path)
        return cap, os.path.basename(resolved_path) if is_file else source_str, is_file

    return None, f"Could not open source: {source_str}", False


# ──────────────────────────────────────────────
# Pipeline Thread
# ──────────────────────────────────────────────

class PipelineThread(threading.Thread):
    """
    Background worker running BYTETracker + ANPR + FRS + MotionAlert on GPU or CPU.
    """

    def __init__(self, config: dict, stream_state: LiveStreamState):
        super().__init__(daemon=True, name="PipelineThread")
        self.config = config
        self.state = stream_state
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            err_msg = f"Pipeline error: {e}"
            logger.error(err_msg)
            import traceback
            traceback.print_exc()
            self.state.update_stats(is_running=False, error_message=err_msg)
        finally:
            self.state.update_stats(is_running=False)

    def _run_pipeline(self):
        cfg = self.config
        source = cfg.get("source", "1")

        # Open video capture
        cap, source_label, is_video_file = open_video_capture(source)
        if cap is None or not cap.isOpened():
            err = f"Failed to open video source '{source}'. If using webcam, try camera index 1 or 2."
            logger.warning(err)
            self.state.update_stats(is_running=False, error_message=err)
            return

        # Configure GPU / CPU device
        req_device = str(cfg.get("device", "gpu")).lower()
        if req_device in ("gpu", "cuda") and torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = f"GPU ({torch.cuda.get_device_name(0)})"
            fp16_enabled = cfg.get("fp16", True)
        else:
            device = torch.device("cpu")
            device_name = "CPU"
            fp16_enabled = False

        self.state.update_stats(
            is_running=True,
            source=source_label,
            device=device_name,
            error_message="",
        )
        logger.info(f"[PipelineThread] Running on {device_name} (FP16={fp16_enabled}) for source: {source_label}")

        import argparse
        args = argparse.Namespace(
            track_thresh=cfg.get("track_thresh", 0.25),
            track_buffer=cfg.get("track_buffer", 30),
            match_thresh=cfg.get("match_thresh", 0.8),
            aspect_ratio_thresh=cfg.get("aspect_ratio_thresh", 1.6),
            min_box_area=cfg.get("min_box_area", 10),
            mot20=False,
            detector_mode=cfg.get("detector_mode", "single_class_test"),
            class_names="HUMAN,VEHICLE",
        )

        # Resolve paths relative to ByteTrack root
        exp_file = cfg.get("exp_file", "exps/example/mot/yolox_x_mix_det.py")
        ckpt_file = cfg.get("ckpt_file", "pretrained/bytetrack_x_mot17.pth.tar")
        if not os.path.isabs(exp_file):
            exp_file = os.path.join(_bytetrack_root, exp_file)
        if not os.path.isabs(ckpt_file):
            ckpt_file = os.path.join(_bytetrack_root, ckpt_file)

        # Resolve DB paths
        frs_db = cfg.get("frs_db", "frs_faces.db")
        anpr_db = cfg.get("anpr_db", "anpr_watchlist.db")
        if not os.path.isabs(frs_db):
            frs_db = os.path.join(_bytetrack_root, frs_db)
        if not os.path.isabs(anpr_db):
            anpr_db = os.path.join(_bytetrack_root, anpr_db)

        # Load YOLOX Detector
        predictor = None
        exp = None
        try:
            exp = get_exp(exp_file, None)
            model = exp.get_model().to(device)
            model.eval()
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model", ckpt))
            if fp16_enabled and device.type == "cuda":
                model.half()
            predictor = Predictor(model, exp, device=device, fp16=(fp16_enabled and device.type == "cuda"))
            logger.info(f"[PipelineThread] YOLOX loaded on {device}")
        except Exception as e:
            logger.warning(f"[PipelineThread] Model load fallback ({e}). Camera-only mode active.")

        tracker = BYTETracker(args, frame_rate=30) if predictor else None
        track_router = TrackRouter(
            class_names=["HUMAN", "VEHICLE"],
            default_mode=args.detector_mode,
        )

        enable_alert = cfg.get("enable_motion_alert", True)
        alert_system = MotionAlertSystem(
            low_thresh=cfg.get("alert_low", 20.0),
            med_thresh=cfg.get("alert_med", 60.0),
            high_thresh=cfg.get("alert_high", 120.0),
            enable_sound=False,
        ) if enable_alert else None

        enable_frs = cfg.get("enable_frs", False)
        frs_pipeline = FRSPipeline(
            db_path=frs_db,
            min_box_area=cfg.get("frs_min_area", 1500.0),
            num_workers=cfg.get("frs_workers", 1),
            match_threshold=cfg.get("frs_threshold", 0.60),
        ) if enable_frs else None

        enable_anpr = cfg.get("enable_anpr", False)
        anpr_pipeline = ANPRPipeline(
            db_path=anpr_db,
            min_box_area=cfg.get("anpr_min_area", 800.0),
            num_workers=cfg.get("anpr_workers", 1),
        ) if enable_anpr else None

        timer = Timer()
        frame_id = 0
        frs_alert_count = 0
        anpr_alert_count = 0
        ALERT_COLORS = {
            "NORMAL": "#22c55e", "LOW": "#facc15",
            "MEDIUM": "#f97316", "HIGH": "#dc2626",
        }
        _last_outputs = None
        _last_img_info = None
        _last_valid_targets: list = []
        detect_skip = max(1, int(cfg.get("detect_skip", 1)))
        retry_drops = 0

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    retry_drops += 1
                    time.sleep(0.04)
                    if retry_drops > 40:
                        logger.warning(f"Video capture dropped repeatedly for source: {source}")
                        break
                    continue

            retry_drops = 0
            frame_id += 1
            valid_targets: list = []
            human_count = 0
            vehicle_count = 0
            current_alert = "NORMAL"

            if predictor is None:
                # Camera-only fallback mode
                online_im = frame.copy()
                cv2.putText(
                    online_im, "LIVE CAMERA FEED (MODEL NOT LOADED)",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2,
                )
            else:
                run_detection = (frame_id % detect_skip == 0)

                if run_detection:
                    timer.tic()
                    outputs, img_info = predictor.inference(frame, timer)
                    _last_outputs = outputs
                    _last_img_info = img_info
                    timer.toc()
                else:
                    outputs = _last_outputs
                    if _last_img_info is not None:
                        img_info = dict(_last_img_info)
                        img_info["raw_img"] = frame
                    else:
                        img_info = {
                            "raw_img": frame,
                            "height": frame.shape[0],
                            "width": frame.shape[1],
                        }

                frs_results = None
                anpr_results = None
                alert_data = None

                if outputs is not None and outputs[0] is not None and run_detection:
                    online_targets = tracker.update(
                        outputs[0],
                        [img_info["height"], img_info["width"]],
                        exp.test_size,
                    )
                    for t in online_targets:
                        if t.tlwh[2] * t.tlwh[3] > args.min_box_area:
                            valid_targets.append(t)
                    _last_valid_targets = valid_targets
                else:
                    valid_targets = _last_valid_targets

                routing_result = track_router.route(
                    valid_targets, detector_mode=args.detector_mode
                )
                human_count = len(routing_result.human_tracks)
                vehicle_count = len(routing_result.vehicle_tracks)

                # Motion Alert update
                if alert_system is not None:
                    all_tlwhs = [t.tlwh for t in valid_targets]
                    all_ids = [t.track_id for t in valid_targets]
                    alert_data = alert_system.update(all_tlwhs, all_ids)
                    if alert_data:
                        levels = [v.get("level", "NORMAL") for v in alert_data.values()]
                        if "HIGH" in levels:
                            current_alert = "HIGH"
                        elif "MEDIUM" in levels:
                            current_alert = "MEDIUM"
                        elif "LOW" in levels:
                            current_alert = "LOW"

                # FRS execution
                if frs_pipeline is not None:
                    hum_tlwhs = [t.tlwh for t in routing_result.human_tracks]
                    hum_ids = [t.track_id for t in routing_result.human_tracks]
                    hum_scores = [t.score for t in routing_result.human_tracks]
                    frs_results = frs_pipeline.process_frame(
                        frame, hum_tlwhs, hum_ids, hum_scores, current_time=time.time()
                    )
                    if frs_results:
                        for tid, res in frs_results.items():
                            if res.get("is_flagged"):
                                frs_alert_count += 1
                                self.state.update_stats(
                                    last_frs_match={
                                        "name": res.get("name", "Unknown"),
                                        "person_id": res.get("person_id", ""),
                                        "confidence": round(res.get("confidence", 0.0), 3),
                                        "category": res.get("category", ""),
                                        "timestamp": time.strftime("%H:%M:%S"),
                                    },
                                    frs_alerts_session=frs_alert_count,
                                )

                # ANPR execution
                if anpr_pipeline is not None:
                    veh_tlwhs = [t.tlwh for t in routing_result.vehicle_tracks]
                    veh_ids = [t.track_id for t in routing_result.vehicle_tracks]
                    veh_scores = [t.score for t in routing_result.vehicle_tracks]
                    anpr_results = anpr_pipeline.process_frame(
                        frame, veh_tlwhs, veh_ids, veh_scores, current_time=time.time()
                    )
                    if anpr_results:
                        for tid, res in anpr_results.items():
                            if res.get("is_flagged"):
                                anpr_alert_count += 1
                                self.state.update_stats(
                                    last_anpr_match={
                                        "plate": res.get("plate_number", ""),
                                        "category": res.get("alert_category", ""),
                                        "confidence": round(res.get("confidence", 0.0), 3),
                                        "timestamp": time.strftime("%H:%M:%S"),
                                    },
                                    anpr_alerts_session=anpr_alert_count,
                                )

                # Visualization HUD
                all_tlwhs = [t.tlwh for t in valid_targets]
                all_ids = [t.track_id for t in valid_targets]
                fps_val = 1.0 / max(1e-5, timer.average_time)

                if frs_pipeline is not None and anpr_pipeline is not None:
                    online_im = track_router.draw_unified_overlay(
                        frame, routing_result,
                        anpr_results=anpr_results or {},
                        frs_results=frs_results or {},
                        alert_data=alert_data,
                        detector_mode=args.detector_mode,
                        frame_id=frame_id, fps=fps_val,
                    )
                elif frs_pipeline is not None:
                    online_im = FRSVisualizer.draw_frs_overlay(
                        frame, all_tlwhs, all_ids,
                        frs_results or {}, frame_id=frame_id, fps=fps_val,
                    )
                elif anpr_pipeline is not None:
                    online_im = ANPRVisualizer.draw_anpr_overlay(
                        frame, all_tlwhs, all_ids,
                        anpr_results or {}, frame_id=frame_id, fps=fps_val,
                    )
                else:
                    online_im = plot_tracking(
                        frame, all_tlwhs, all_ids,
                        frame_id=frame_id, fps=fps_val,
                    )

                # Motion alert HUD badge
                if current_alert != "NORMAL":
                    badge_color = {
                        "LOW": (0, 255, 255),
                        "MEDIUM": (0, 128, 255),
                        "HIGH": (0, 0, 255),
                    }[current_alert]
                    h, w = online_im.shape[:2]
                    cv2.rectangle(online_im, (w - 210, 8), (w - 8, 44), badge_color, -1)
                    cv2.putText(
                        online_im, f"MOTION: {current_alert}", (w - 200, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    )

            # Draw Device Info HUD badge (bottom left)
            dev_badge = f"{device_name} | FP16:{'ON' if fp16_enabled else 'OFF'}"
            cv2.putText(
                online_im, dev_badge, (12, online_im.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA,
            )

            # JPEG Encode & Broadcast
            _, jpeg = cv2.imencode(".jpg", online_im, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.state.set_frame(jpeg.tobytes())
            self.state.update_stats(
                fps=1.0 / max(1e-5, timer.average_time) if predictor else 30.0,
                frame_id=frame_id,
                active_tracks=len(valid_targets),
                human_count=human_count,
                vehicle_count=vehicle_count,
                motion_alert_level=current_alert,
                motion_alert_color=ALERT_COLORS.get(current_alert, "#22c55e"),
            )

            if predictor is None:
                time.sleep(0.033)

        # Cleanup
        cap.release()
        if frs_pipeline:
            frs_pipeline.stop()
        if anpr_pipeline:
            anpr_pipeline.stop()
        self.state.update_stats(is_running=False)
        logger.info("[PipelineThread] Cleanly terminated.")


# ──────────────────────────────────────────────
# MJPEG Generator
# ──────────────────────────────────────────────

async def mjpeg_generator(stream_state: LiveStreamState):
    """Async generator yielding multipart/x-mixed-replace JPEG frames."""
    while True:
        frame = stream_state.get_frame()
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        await asyncio.sleep(0.033)  # ~30 FPS cap

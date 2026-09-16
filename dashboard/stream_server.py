"""
Enterprise-Grade Multi-Camera Ingestion & Tracking Engine for ByteTrack Surveillance Dashboard.
Supports:
  - Cluster of smartphones running DroidCam (http://<IP>:4747/video)
  - IP CCTV cameras via RTSP (rtsp://...)
  - Local USB / laptop webcams via DirectShow (indices "0", "1", "2")
  - Looping pre-recorded video files
Features:
  - Zero-lag decoupled frame grabbing (discards accumulated OS socket buffers)
  - Isolated BYTETracker, MotionAlert, and TrackRouter states per camera
  - Resource-efficient shared YOLOX GPU/CPU inference engine
  - Multi-camera orchestration and MJPEG broadcasting
"""

import os
import sys
import time
import threading
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

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
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.alert_system import MotionAlertSystem
from yolox.tracker.persistent_tracker import GLOBAL_PERSISTENT_TRACKER
from yolox.anpr import ANPRPipeline, ANPRVisualizer
from yolox.frs import FRSPipeline, FRSVisualizer
from yolox.routing import TrackRouter
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer
from demo_track import Predictor
from dashboard.camera_manager import CAMERA_REGISTRY


# ──────────────────────────────────────────────
# Telemetry & Camera Stream State
# ──────────────────────────────────────────────

@dataclass
class CameraStats:
    cam_id: int = 0
    name: str = "CAM"
    location: str = "Surveillance Zone"
    is_running: bool = False
    status: str = "OFFLINE"  # "LIVE", "CONNECTING", "OFFLINE", "ERROR"
    source: str = ""
    device: str = "GPU"
    fps: float = 0.0
    frame_id: int = 0
    active_tracks: int = 0
    human_count: int = 0
    vehicle_count: int = 0
    motion_alert_level: str = "NORMAL"  # "NORMAL", "LOW", "MEDIUM", "HIGH"
    motion_alert_color: str = "#22c55e"
    error_message: str = ""
    last_frs_match: Optional[Dict[str, Any]] = None
    last_anpr_match: Optional[Dict[str, Any]] = None
    frs_alerts_session: int = 0
    anpr_alerts_session: int = 0


def create_placeholder_jpeg(name: str = "Camera", status: str = "OFFLINE", width: int = 640, height: int = 360, detail: str = "") -> bytes:
    """Generate a clean dark placeholder frame for offline / connecting cameras with diagnostic detail."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Dark subtle gradient background
    cv2.rectangle(img, (0, 0), (width, height), (15, 23, 42), -1)
    # Border
    cv2.rectangle(img, (2, 2), (width - 2, height - 2), (51, 65, 85), 2)
    # Title
    cv2.putText(img, name.upper(), (30, height // 2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (241, 245, 249), 2)
    # Status
    color = (34, 197, 94) if "LIVE" in status else ((245, 158, 11) if "CONNECTING" in status else ((239, 68, 68) if "ERROR" in status or "BUSY" in status else (100, 116, 139)))
    cv2.putText(img, f"STATUS: {status}", (30, height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    # Detail / diagnostics
    if detail:
        cv2.putText(img, str(detail)[:70], (30, height // 2 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (248, 113, 113), 1)
    cv2.putText(img, "IBVAP Multi-Camera Ingestion Cluster", (30, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1)
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()



class CameraStreamState:
    """Thread-safe state container for a single camera stream."""

    def __init__(self, cam_id: int = 0, name: str = "CAM", location: str = ""):
        self._lock = threading.Lock()
        self.stats = CameraStats(cam_id=cam_id, name=name, location=location)
        self._frame: Optional[bytes] = create_placeholder_jpeg(name, "OFFLINE")

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
                "cam_id": s.cam_id,
                "name": s.name,
                "location": s.location,
                "is_running": s.is_running,
                "status": s.status,
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
            name = self.stats.name
            loc = self.stats.location
            cid = self.stats.cam_id
            self.stats = CameraStats(cam_id=cid, name=name, location=loc)
            self._frame = create_placeholder_jpeg(name, "OFFLINE")


# Backward compatibility alias
LiveStreamState = CameraStreamState


# ──────────────────────────────────────────────
# Zero-Lag Decoupled Video Capture
# ──────────────────────────────────────────────

class ZeroLagCapture:
    """
    Decoupled non-blocking video reader thread.
    Continuously drains frames from the underlying VideoCapture buffer,
    keeping only the single most recent frame. This eliminates video latency
    for RTSP and DroidCam HTTP streams regardless of inference time.
    """

    def __init__(self, source_str: str, max_reconnects: int = 5):
        self.source_str = str(source_str).strip()
        self.max_reconnects = max_reconnects
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.source_label = ""
        self.error_label = ""
        self.is_video_file = False
        self.is_opened = False
        self.frame_count = 0

    def start(self) -> bool:
        cap, label, is_file = self._open_source()
        if cap is None or not cap.isOpened():
            self.error_label = label
            return False

        self._cap = cap
        self.source_label = label
        self.is_video_file = is_file
        self.is_opened = True

        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name=f"ZeroLagCap-{self.source_label}")
        self._thread.start()
        return True

    def _open_source(self) -> Tuple[Optional[cv2.VideoCapture], str, bool]:
        src = self.source_str

        # 1. Numeric webcam index (e.g. "0", "1", "2")
        if src.isdigit():
            cam_idx = int(src)
            # Find backend yielding the highest resolution (MSMF gives 720p HD on modern Windows laptop webcams)
            best_cap = None
            best_label = ""
            best_w = 0
            backends = [
                (cv2.CAP_MSMF, f"Webcam {cam_idx} (MSMF)"),
                (cv2.CAP_DSHOW, f"Webcam {cam_idx} (DSHOW)"),
                (cv2.CAP_ANY, f"Webcam {cam_idx}"),
            ]
            for backend, label in backends:
                try:
                    cap = cv2.VideoCapture(cam_idx, backend)
                    if cap.isOpened():
                        try:
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        except Exception:
                            pass
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        if w > best_w:
                            if best_cap is not None:
                                best_cap.release()
                            best_cap = cap
                            best_label = label
                            best_w = w
                            if w >= 1280:
                                break
                        else:
                            cap.release()
                except Exception:
                    pass

            if best_cap is not None and best_cap.isOpened():
                logger.info(f"[ZeroLagCapture] Selected {best_label} @ {best_w}px width")
                return best_cap, best_label, False

            return None, f"Webcam index {cam_idx} unavailable", False

        # 2. DroidCam HTTP MJPEG stream (e.g. http://10.131.46.15:4747/video)
        # 3. RTSP stream (e.g. rtsp://192.168.1.100:554/live)
        if src.startswith("http://") or src.startswith("https://") or src.startswith("rtsp://"):
            # Check if DroidCam phone is currently connected to PC app
            if ":4747" in src:
                try:
                    import urllib.request
                    req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=1.5) as resp:
                        body_prefix = resp.read(512).decode("utf-8", errors="ignore").lower()
                        if "busy" in body_prefix or "connected to the pc client" in body_prefix:
                            err = "DroidCam phone is busy: connected to PC Client. Close DroidCamApp or use Webcam '2'."
                            logger.warning(f"[ZeroLagCapture] {err}")
                            return None, err, False
                except Exception:
                    pass

            candidate_urls = [src]
            if ":4747" in src and src.endswith("/video"):
                candidate_urls.append(src.replace("/video", "/mjpegfeed"))

            for candidate in candidate_urls:
                try:
                    cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        label = "DroidCam Stream" if ":4747" in candidate else ("RTSP Cam" if "rtsp://" in candidate else "Network Stream")
                        return cap, f"{label} ({candidate})", False
                except Exception as e:
                    logger.warning(f"[ZeroLagCapture] Error opening network stream {candidate}: {e}")

                try:
                    cap = cv2.VideoCapture(candidate)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        return cap, candidate, False
                except Exception:
                    pass

            return None, f"Could not connect to stream: {src}", False

        # 4. Local video file path
        resolved = src
        if not os.path.isabs(src):
            candidate = os.path.join(_bytetrack_root, src)
            if os.path.isfile(candidate):
                resolved = candidate

        if os.path.isfile(resolved):
            cap = cv2.VideoCapture(resolved)
            if cap.isOpened():
                return cap, os.path.basename(resolved), True

        return None, f"Source not found: {src}", False

    def _reader_loop(self):
        consecutive_errors = 0
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self.is_video_file:
                    # Loop video continuously
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    consecutive_errors += 1
                    time.sleep(0.05)
                    if consecutive_errors > 30 and not self.is_video_file:
                        # Attempt stream reconnection for DroidCam / RTSP
                        logger.warning(f"[ZeroLagCapture] Stream dropped: {self.source_label}. Attempting reconnect...")
                        try:
                            self._cap.release()
                        except Exception:
                            pass
                        time.sleep(1.0)
                        cap, _, _ = self._open_source()
                        if cap and cap.isOpened():
                            self._cap = cap
                            consecutive_errors = 0
                            logger.info(f"[ZeroLagCapture] Stream reconnected: {self.source_label}")
                    continue

            consecutive_errors = 0
            self.frame_count += 1

            # Discard any prior frame and update single slot under lock
            with self._frame_lock:
                self._latest_frame = frame

            # Regulate loop speed for video files to ~30 FPS, unthrottled for live cameras
            if self.is_video_file:
                time.sleep(0.025)
            else:
                time.sleep(0.002)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieve the newest available frame without waiting or lagging."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
        return False, None

    def stop(self):
        self._stop_event.set()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self.is_opened = False


# ──────────────────────────────────────────────
# Shared Neural Network Detector (CUDA / CPU)
# ──────────────────────────────────────────────

class SharedDetector:
    """
    Thread-safe shared YOLOX detector singleton.
    Enables multiple concurrent cameras to execute inference on a single GPU model
    without duplicating VRAM usage or causing CUDA memory exhaustion.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.predictor: Optional[Predictor] = None
        self.exp = None
        self.device = torch.device("cpu")
        self.device_name = "CPU"
        self.fp16_enabled = False
        self._loaded = False

    def load(self, cfg: dict):
        with self._lock:
            if self._loaded and self.predictor is not None:
                return

            req_device = str(cfg.get("device", "gpu")).lower()
            if req_device in ("gpu", "cuda") and torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.device_name = f"GPU ({torch.cuda.get_device_name(0)})"
                self.fp16_enabled = cfg.get("fp16", True)
            else:
                self.device = torch.device("cpu")
                self.device_name = "CPU"
                self.fp16_enabled = False

            exp_file = cfg.get("exp_file", "exps/example/mot/yolox_x_mix_det.py")
            ckpt_file = cfg.get("ckpt_file", "pretrained/bytetrack_x_mot17.pth.tar")
            if not os.path.isabs(exp_file):
                exp_file = os.path.join(_bytetrack_root, exp_file)
            if not os.path.isabs(ckpt_file):
                ckpt_file = os.path.join(_bytetrack_root, ckpt_file)

            try:
                self.exp = get_exp(exp_file, None)
                model = self.exp.get_model().to(self.device)
                model.eval()
                ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt.get("model", ckpt))
                if self.fp16_enabled and self.device.type == "cuda":
                    model.half()
                self.predictor = Predictor(
                    model, self.exp, device=self.device,
                    fp16=(self.fp16_enabled and self.device.type == "cuda"),
                )
                self._loaded = True
                logger.info(f"[SharedDetector] YOLOX loaded on {self.device_name} (FP16={self.fp16_enabled})")
            except Exception as e:
                logger.warning(f"[SharedDetector] Model load failed ({e}). Camera-only mode active.")
                self.predictor = None
                self._loaded = False

    def inference(self, frame: np.ndarray, timer: Timer):
        with self._lock:
            if self.predictor is None:
                return None, None
            return self.predictor.inference(frame, timer)


SHARED_DETECTOR = SharedDetector()


# ──────────────────────────────────────────────
# Per-Camera Pipeline Worker Thread
# ──────────────────────────────────────────────

class CameraPipelineWorker(threading.Thread):
    """
    Dedicated background worker for an individual camera stream.
    Maintains isolated BYTETracker, MotionAlertSystem, TrackRouter,
    and FRS/ANPR tracking state to prevent cross-camera ID collision.
    """

    def __init__(self, cam_info: dict, pipeline_cfg: dict, stream_state: CameraStreamState):
        cam_id = cam_info.get("id", 0)
        cam_name = cam_info.get("name", f"CAM {cam_id:02d}")
        super().__init__(daemon=True, name=f"CamWorker-{cam_id}")
        self.cam_info = cam_info
        self.cam_id = cam_id
        self.cam_name = cam_name
        self.location = cam_info.get("location", "")
        self.source_url = cam_info.get("url", "0")
        self.cfg = pipeline_cfg
        self.state = stream_state
        self.cap: Optional[ZeroLagCapture] = None
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        if self.cap is not None:
            self.cap.stop()

    def run(self):
        try:
            self._run_loop()
        except Exception as e:
            err_msg = f"Camera {self.cam_id} pipeline error: {e}"
            logger.error(err_msg)
            import traceback
            traceback.print_exc()
            self.state.update_stats(is_running=False, status="ERROR", error_message=err_msg)
        finally:
            self.state.update_stats(is_running=False, status="OFFLINE", fps=0.0)

    def _run_loop(self):
        self.state.update_stats(is_running=True, status="CONNECTING", error_message="")
        logger.info(f"[CamWorker-{self.cam_id}] Starting capture for {self.cam_name} from: {self.source_url}")

        # Start decoupled Zero-Lag Capture
        self.cap = ZeroLagCapture(self.source_url)
        if not self.cap.start():
            err = getattr(self.cap, 'error_label', '') or f"Failed to connect: {self.source_url}"
            logger.warning(f"[CamWorker-{self.cam_id}] {err}")
            self.state.update_stats(is_running=False, status="ERROR", error_message=err)
            self.state.set_frame(create_placeholder_jpeg(self.cam_name, "ERROR", detail=err))
            return

        # Ensure shared detector is loaded
        SHARED_DETECTOR.load(self.cfg)

        self.state.update_stats(
            is_running=True,
            status="LIVE",
            source=self.cap.source_label,
            device=SHARED_DETECTOR.device_name,
            error_message="",
        )

        import argparse
        args = argparse.Namespace(
            track_thresh=self.cfg.get("track_thresh", 0.25),
            track_buffer=self.cfg.get("track_buffer", 30),
            match_thresh=self.cfg.get("match_thresh", 0.8),
            aspect_ratio_thresh=self.cfg.get("aspect_ratio_thresh", 1.6),
            min_box_area=self.cfg.get("min_box_area", 10),
            mot20=False,
            detector_mode=self.cfg.get("detector_mode", "single_class_test"),
            class_names="HUMAN,VEHICLE",
        )

        # ISOLATED TRACKING & ROUTING INSTANCES FOR THIS CAMERA
        tracker = BYTETracker(args, frame_rate=30)
        track_router = TrackRouter(class_names=["HUMAN", "VEHICLE"], default_mode=args.detector_mode)

        enable_alert = self.cfg.get("enable_motion_alert", True)
        alert_system = MotionAlertSystem(
            low_thresh=self.cfg.get("alert_low", 20.0),
            med_thresh=self.cfg.get("alert_med", 60.0),
            high_thresh=self.cfg.get("alert_high", 120.0),
            enable_sound=False,
        ) if enable_alert else None

        # Resolve DB paths
        frs_db = self.cfg.get("frs_db", "frs_faces.db")
        anpr_db = self.cfg.get("anpr_db", "anpr_watchlist.db")
        if not os.path.isabs(frs_db):
            frs_db = os.path.join(_bytetrack_root, frs_db)
        if not os.path.isabs(anpr_db):
            anpr_db = os.path.join(_bytetrack_root, anpr_db)

        enable_frs = self.cfg.get("enable_frs", False)
        frs_pipeline = FRSPipeline(
            db_path=frs_db,
            min_box_area=self.cfg.get("frs_min_area", 1500.0),
            num_workers=self.cfg.get("frs_workers", 1),
            match_threshold=self.cfg.get("frs_threshold", 0.60),
        ) if enable_frs else None

        enable_anpr = self.cfg.get("enable_anpr", False)
        anpr_pipeline = ANPRPipeline(
            db_path=anpr_db,
            min_box_area=self.cfg.get("anpr_min_area", 800.0),
            num_workers=self.cfg.get("anpr_workers", 1),
        ) if enable_anpr else None

        timer = Timer()
        frame_id = 0
        frs_alert_count = 0
        anpr_alert_count = 0
        detect_skip = max(1, int(self.cfg.get("detect_skip", 2)))
        _last_outputs = None
        _last_img_info = None
        _last_valid_targets: list = []
        last_frame_time = time.time()
        smooth_fps = 30.0

        ALERT_COLORS = {
            "NORMAL": "#22c55e", "LOW": "#facc15",
            "MEDIUM": "#f97316", "HIGH": "#dc2626",
        }

        logger.info(f"[CamWorker-{self.cam_id}] Pipeline active for {self.cam_name}")

        while not self._stop_event.is_set():
            if self.cap is None:
                break
            ret, frame = self.cap.read_latest()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_id += 1
            now_t = time.time()
            dt = now_t - last_frame_time
            last_frame_time = now_t
            if dt > 0:
                instant_fps = 1.0 / max(1e-4, dt)
                smooth_fps = 0.85 * smooth_fps + 0.15 * instant_fps
            valid_targets: list = []
            human_count = 0
            vehicle_count = 0
            current_alert = "NORMAL"

            if SHARED_DETECTOR.predictor is None:
                # Camera-only fallback mode
                online_im = frame.copy()
                cv2.putText(
                    online_im, f"{self.cam_name} (MODEL NOT LOADED)",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2,
                )
            else:
                run_detection = (frame_id % detect_skip == 0)

                if run_detection:
                    timer.tic()
                    outputs, img_info = SHARED_DETECTOR.inference(frame, timer)
                    timer.toc()
                    _last_outputs = outputs
                    _last_img_info = img_info
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
                        SHARED_DETECTOR.exp.test_size,
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

                # Persistent ID allocation for every tracked target
                human_ids_set = {t.track_id for t in routing_result.human_tracks}
                vehicle_ids_set = {t.track_id for t in routing_result.vehicle_tracks}
                for t in valid_targets:
                    etype = "HUMAN" if t.track_id in human_ids_set else ("VEHICLE" if t.track_id in vehicle_ids_set else "OBJECT")
                    spd = alert_data.get(t.track_id, {}).get("speed", 0.0) if alert_data else 0.0
                    t.persistent_id = GLOBAL_PERSISTENT_TRACKER.assign_or_get_id(
                        cam_id=self.cam_id,
                        cam_name=self.cam_name,
                        local_track_id=t.track_id,
                        entity_type=etype,
                        tlwh=t.tlwh,
                        frame=frame,
                        speed=spd,
                    )

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
                            is_id = bool(res.get("person_id") and res.get("person_id") != "UNKNOWN")
                            GLOBAL_PERSISTENT_TRACKER.update_identification(
                                cam_id=self.cam_id,
                                local_track_id=tid,
                                is_identified=is_id,
                                identified_id=res.get("person_id", "UNIDENTIFIED"),
                                identified_name=res.get("name", "Unidentified Person"),
                                category=res.get("category", "NORMAL"),
                                confidence=res.get("confidence", 0.0),
                                cam_name=self.cam_name,
                            )
                            if res.get("is_flagged"):
                                frs_alert_count += 1
                                self.state.update_stats(
                                    last_frs_match={
                                        "cam_id": self.cam_id,
                                        "cam_name": self.cam_name,
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
                            plate = res.get("plate_number")
                            is_id = bool(plate)
                            GLOBAL_PERSISTENT_TRACKER.update_identification(
                                cam_id=self.cam_id,
                                local_track_id=tid,
                                is_identified=is_id,
                                identified_id=plate or "UNIDENTIFIED",
                                identified_name=f"Vehicle {plate}" if plate else "Unidentified Vehicle",
                                category=res.get("alert_category", "NORMAL"),
                                confidence=res.get("confidence", 0.0),
                                cam_name=self.cam_name,
                            )
                            if res.get("is_flagged"):
                                anpr_alert_count += 1
                                self.state.update_stats(
                                    last_anpr_match={
                                        "cam_id": self.cam_id,
                                        "cam_name": self.cam_name,
                                        "plate": res.get("plate_number", ""),
                                        "category": res.get("alert_category", ""),
                                        "confidence": round(res.get("confidence", 0.0), 3),
                                        "timestamp": time.strftime("%H:%M:%S"),
                                    },
                                    anpr_alerts_session=anpr_alert_count,
                                )

                # Visualization: Draw clean bounding boxes, labels, and persistent IDs without clashing top headers
                fps_val = 1.0 / max(1e-5, timer.average_time)
                online_im = track_router.draw_unified_overlay(
                    frame, routing_result,
                    anpr_results=anpr_results or {},
                    frs_results=frs_results or {},
                    alert_data=alert_data,
                    detector_mode=args.detector_mode,
                    frame_id=frame_id, fps=fps_val,
                    show_header=False,
                )

            # Web preview scaling: maintain crisp 720p HD, only downsample 4K/1080p > 1280 using INTER_AREA
            h_im, w_im = online_im.shape[:2]
            if w_im > 1280:
                scale_w = 1280.0 / w_im
                preview_im = cv2.resize(online_im, (1280, int(h_im * scale_w)), interpolation=cv2.INTER_AREA)
            else:
                preview_im = online_im

            # High-clarity JPEG encoding (Quality 88 preserves facial details and sharp text without blockiness)
            _, jpeg = cv2.imencode(".jpg", preview_im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            self.state.set_frame(jpeg.tobytes())
            self.state.update_stats(
                fps=round(smooth_fps, 1),
                frame_id=frame_id,
                active_tracks=len(valid_targets),
                human_count=human_count,
                vehicle_count=vehicle_count,
                motion_alert_level=current_alert,
                motion_alert_color=ALERT_COLORS.get(current_alert, "#22c55e"),
            )

            # Responsive yield
            time.sleep(0.004)

        # Cleanup
        if self.cap is not None:
            self.cap.stop()
        if frs_pipeline:
            frs_pipeline.stop()
        if anpr_pipeline:
            anpr_pipeline.stop()
        self.state.update_stats(is_running=False, status="OFFLINE", fps=0.0)
        self.state.set_frame(create_placeholder_jpeg(self.cam_name, "OFFLINE"))
        logger.info(f"[CamWorker-{self.cam_id}] Stopped cleanly.")


# ──────────────────────────────────────────────
# Multi-Camera Cluster Manager
# ──────────────────────────────────────────────

class MultiCameraManager:
    """
    Central orchestrator for multi-camera cluster.
    Coordinates worker threads, states, and streams across all cameras.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.workers: Dict[int, CameraPipelineWorker] = {}
        self.states: Dict[int, CameraStreamState] = {}
        self._default_cfg: Dict[str, Any] = {
            "device": "gpu",
            "fp16": True,
            "enable_frs": False,
            "enable_anpr": False,
            "enable_motion_alert": True,
            "detect_skip": 1,
            "exp_file": "exps/example/mot/yolox_x_mix_det.py",
            "ckpt_file": "pretrained/bytetrack_x_mot17.pth.tar",
            "frs_db": "frs_faces.db",
            "anpr_db": "anpr_watchlist.db",
            "frs_min_area": 1500.0,
            "frs_threshold": 0.60,
            "frs_workers": 1,
            "anpr_min_area": 800.0,
            "anpr_workers": 1,
            "track_thresh": 0.25,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "detector_mode": "single_class_test",
            "alert_low": 20.0,
            "alert_med": 60.0,
            "alert_high": 120.0,
        }
        # Initialize camera states from registry
        self._sync_states_from_registry()

    def _sync_states_from_registry(self):
        cams = CAMERA_REGISTRY.load_cameras()
        with self._lock:
            for c in cams:
                cid = c.get("id", 0)
                if cid not in self.states:
                    self.states[cid] = CameraStreamState(
                        cam_id=cid,
                        name=c.get("name", f"CAM {cid:02d}"),
                        location=c.get("location", ""),
                    )
                else:
                    self.states[cid].update_stats(
                        name=c.get("name", f"CAM {cid:02d}"),
                        location=c.get("location", ""),
                    )

    def get_state(self, cam_id: int) -> CameraStreamState:
        with self._lock:
            if cam_id not in self.states:
                cam = CAMERA_REGISTRY.get_camera(cam_id)
                name = cam.get("name", f"CAM {cam_id:02d}") if cam else f"CAM {cam_id:02d}"
                loc = cam.get("location", "") if cam else ""
                self.states[cam_id] = CameraStreamState(cam_id=cam_id, name=name, location=loc)
            return self.states[cam_id]

    def start_camera(self, cam_id: int, pipeline_cfg: Optional[dict] = None) -> bool:
        """Start streaming pipeline for an individual camera."""
        self._sync_states_from_registry()
        cam = CAMERA_REGISTRY.get_camera(cam_id)
        if not cam:
            logger.error(f"[MultiCameraManager] Camera {cam_id} not found in registry")
            return False

        with self._lock:
            # Stop existing worker if active
            if cam_id in self.workers and self.workers[cam_id].is_alive():
                self.workers[cam_id].stop()
                self.workers[cam_id].join(timeout=2)

            cfg = dict(self._default_cfg)
            if pipeline_cfg:
                cfg.update(pipeline_cfg)

            state = self.get_state(cam_id)
            state.reset()
            worker = CameraPipelineWorker(cam, cfg, state)
            self.workers[cam_id] = worker
            worker.start()
            logger.info(f"[MultiCameraManager] Camera {cam_id} started.")
            return True

    def stop_camera(self, cam_id: int) -> bool:
        """Stop streaming pipeline for an individual camera."""
        with self._lock:
            if cam_id in self.workers:
                worker = self.workers[cam_id]
                worker.stop()
                del self.workers[cam_id]
                if cam_id in self.states:
                    self.states[cam_id].update_stats(is_running=False, status="OFFLINE", fps=0.0)
                    self.states[cam_id].set_frame(create_placeholder_jpeg(self.states[cam_id].stats.name, "OFFLINE"))
                logger.info(f"[MultiCameraManager] Camera {cam_id} stopped.")
                return True
        return False

    def start_all(self, pipeline_cfg: Optional[dict] = None) -> Dict[int, bool]:
        """Concurrently start all enabled cameras."""
        enabled_cams = CAMERA_REGISTRY.get_enabled_cameras()
        results = {}
        for c in enabled_cams:
            cid = c.get("id")
            if cid is not None:
                results[cid] = self.start_camera(cid, pipeline_cfg)
        return results

    def stop_all(self) -> Dict[int, bool]:
        """Stop all running cameras cleanly."""
        with self._lock:
            active_ids = list(self.workers.keys())

        results = {}
        for cid in active_ids:
            results[cid] = self.stop_camera(cid)
        return results

    def get_all_stats(self) -> Dict[str, Any]:
        """Aggregate telemetry dictionary for all registered cameras."""
        self._sync_states_from_registry()
        cams = CAMERA_REGISTRY.load_cameras()
        cam_stats = []
        total_active_tracks = 0
        total_humans = 0
        total_vehicles = 0
        any_high_alert = False

        for c in cams:
            cid = c.get("id", 0)
            state = self.get_state(cid)
            stat = state.get_stats()
            stat["enabled"] = c.get("enabled", True)
            stat["configured_url"] = c.get("url", "")
            cam_stats.append(stat)

            if stat.get("is_running"):
                total_active_tracks += stat.get("active_tracks", 0)
                total_humans += stat.get("human_count", 0)
                total_vehicles += stat.get("vehicle_count", 0)
                if stat.get("motion_alert_level") == "HIGH":
                    any_high_alert = True

        return {
            "cameras": cam_stats,
            "total_cameras": len(cams),
            "running_cameras": sum(1 for s in cam_stats if s.get("is_running")),
            "cluster_metrics": {
                "active_tracks": total_active_tracks,
                "human_count": total_humans,
                "vehicle_count": total_vehicles,
                "overall_status": "HIGH ALERT" if any_high_alert else "NORMAL",
            }
        }


# Global Multi-Camera Manager instance
MULTI_CAMERA_MANAGER = MultiCameraManager()

# Backward compatibility proxy: STREAM_STATE points to primary camera (0)
STREAM_STATE = MULTI_CAMERA_MANAGER.get_state(0)


# Backward compatibility PipelineThread shim
class PipelineThread(threading.Thread):
    def __init__(self, config: dict, stream_state: Any):
        super().__init__(daemon=True, name="PipelineThreadCompat")
        self.config = config
        self.state = stream_state
        self.cam_id = int(config.get("cam_id", 0))

    def stop(self):
        MULTI_CAMERA_MANAGER.stop_camera(self.cam_id)

    def run(self):
        MULTI_CAMERA_MANAGER.start_camera(self.cam_id, self.config)


# ──────────────────────────────────────────────
# Async MJPEG Generator
# ──────────────────────────────────────────────

async def mjpeg_generator(stream_state: CameraStreamState):
    """Async generator yielding multipart/x-mixed-replace JPEG frames for a camera."""
    while True:
        frame = stream_state.get_frame()
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        if not stream_state.stats.is_running:
            await asyncio.sleep(0.25)
        else:
            await asyncio.sleep(0.033)

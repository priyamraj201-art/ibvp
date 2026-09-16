import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from loguru import logger

from yolox.tracker.byte_tracker import STrack


class DetectorMode:
    SINGLE_CLASS_TEST = "single_class_test"
    MULTI_CLASS_PRODUCTION = "multi_class_production"


class RoutingResult:
    """
    Holds categorized tracks and summary statistics for downstream pipelines.
    """
    def __init__(
        self,
        vehicle_tracks: List[STrack],
        human_tracks: List[STrack],
        other_tracks: List[STrack],
        all_tracks: List[STrack],
        class_summary: Dict[str, int],
    ):
        self.vehicle_tracks = vehicle_tracks
        self.human_tracks = human_tracks
        self.other_tracks = other_tracks
        self.all_tracks = all_tracks
        self.class_summary = class_summary

    @property
    def num_vehicles(self) -> int:
        return len(self.vehicle_tracks)

    @property
    def num_humans(self) -> int:
        return len(self.human_tracks)


class TrackRouter:
    """
    Class-Aware Track Router.
    Routes tracked objects to downstream specialized modules (ANPR vs Motion/Speed Alert)
    based on stabilized, majority-voted semantic class identities.
    """

    VEHICLE_LABELS = {"VEHICLE", "CAR", "TRUCK", "BUS", "MOTORCYCLE", "BICYCLE"}
    HUMAN_LABELS = {"HUMAN", "PERSON", "PEDESTRIAN"}

    COCO_CLASSES = [
        "PERSON", "BICYCLE", "CAR", "MOTORCYCLE", "AIRPLANE", "BUS", "TRAIN", "TRUCK", "BOAT",
        "TRAFFIC LIGHT", "FIRE HYDRANT", "STOP SIGN", "PARKING METER", "BENCH", "BIRD", "CAT",
        "DOG", "HORSE", "SHEEP", "COW", "ELEPHANT", "BEAR", "ZEBRA", "GIRAFFE", "BACKPACK",
        "UMBRELLA", "HANDBAG", "TIE", "SUITCASE", "FRISBEE", "SKIS", "SNOWBOARD", "SPORTS BALL",
        "KITE", "BASEBALL BAT", "BASEBALL GLOVE", "SKATEBOARD", "SURFBOARD", "TENNIS RACKET",
        "BOTTLE", "WINE GLASS", "CUP", "FORK", "KNIFE", "SPOON", "BOWL", "BANANA", "APPLE",
        "SANDWICH", "ORANGE", "BROCCOLI", "CARROT", "HOT DOG", "PIZZA", "DONUT", "CAKE",
        "CHAIR", "COUCH", "POTTED PLANT", "BED", "DINING TABLE", "TOILET", "TV", "LAPTOP",
        "MOUSE", "REMOTE", "KEYBOARD", "CELL PHONE", "MICROWAVE", "OVEN", "TOASTER", "SINK",
        "REFRIGERATOR", "BOOK", "CLOCK", "VASE", "SCISSORS", "TEDDY BEAR", "HAIR DRIER", "TOOTHBRUSH"
    ]

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        stability_window: int = 5,
        default_mode: str = DetectorMode.SINGLE_CLASS_TEST,
    ):
        # Default: Class 0 is HUMAN (matches MOT17 & COCO person), Class 1 is VEHICLE
        self.class_names = class_names or ["HUMAN", "VEHICLE"]
        self.stability_window = stability_window
        self.default_mode = default_mode

    def get_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id].upper()
        if 0 <= class_id < len(self.COCO_CLASSES):
            return self.COCO_CLASSES[class_id]
        return f"CLASS_{class_id}"

    def is_vehicle(self, class_name: str) -> bool:
        return class_name in self.VEHICLE_LABELS

    def is_human(self, class_name: str) -> bool:
        return class_name in self.HUMAN_LABELS

    def route(
        self,
        tracks: List[STrack],
        detector_mode: Optional[str] = None,
    ) -> RoutingResult:
        """
        Split active tracks into class-specific buckets with majority-vote stabilization.
        """
        mode = detector_mode or self.default_mode

        vehicle_tracks: List[STrack] = []
        human_tracks: List[STrack] = []
        other_tracks: List[STrack] = []
        class_summary: Dict[str, int] = {"VEHICLE": 0, "HUMAN": 0, "OTHER": 0}

        for track in tracks:
            smoothed_cls = track.smoothed_cls
            class_name = self.get_class_name(smoothed_cls)

            if mode == DetectorMode.MULTI_CLASS_PRODUCTION:
                if self.is_vehicle(class_name):
                    vehicle_tracks.append(track)
                    class_summary["VEHICLE"] += 1
                elif self.is_human(class_name):
                    human_tracks.append(track)
                    class_summary["HUMAN"] += 1
                else:
                    other_tracks.append(track)
                    class_summary["OTHER"] += 1
            else:
                # SINGLE_CLASS_TEST mode:
                # Backward-compatible testing path: all tracks are available to both pipelines
                vehicle_tracks.append(track)
                human_tracks.append(track)
                class_summary["HUMAN"] += 1

        return RoutingResult(
            vehicle_tracks=vehicle_tracks,
            human_tracks=human_tracks,
            other_tracks=other_tracks,
            all_tracks=tracks,
            class_summary=class_summary,
        )

    def draw_unified_overlay(
        self,
        image: np.ndarray,
        routing_result: RoutingResult,
        anpr_results: Optional[Dict[int, Dict[str, Any]]] = None,
        alert_data: Optional[Dict[int, Dict[str, Any]]] = None,
        frs_results: Optional[Dict[int, Dict[str, Any]]] = None,
        detector_mode: str = DetectorMode.SINGLE_CLASS_TEST,
        frame_id: int = 0,
        fps: float = 0.0,
        show_header: bool = True,
    ) -> np.ndarray:
        """
        Render unified visual overlay displaying:
        - License plate badges on VEHICLES
        - Speed labels, identity badges, and trajectory trails on HUMANS
        - Top unified status banner showing counts by class
        """
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        # 1. Render Human Tracks (Motion Velocity & Alerts & FRS)
        for track in routing_result.human_tracks:
            tid = track.track_id
            tlwh = track.tlwh
            x1, y1, w, h = map(int, tlwh)
            x2, y2 = x1 + w, y1 + h

            info = alert_data.get(tid, {
                "level": "NORMAL",
                "speed": 0.0,
                "color": (0, 220, 0),
                "trail": [],
            }) if alert_data else {
                "level": "NORMAL",
                "speed": 0.0,
                "color": (0, 220, 0),
                "trail": [],
            }
            color = info.get("color", (0, 220, 0))
            trail = info.get("trail", [])
            level = info.get("level", "NORMAL")
            speed = info.get("speed", 0.0)

            # Check FRS identification
            face_info = frs_results.get(tid) if frs_results else None
            is_identified = bool(face_info and face_info.get("person_id") and face_info.get("person_id") != "UNKNOWN")
            is_flagged = bool(face_info and face_info.get("is_flagged", False))

            # Retained Person ID (from Re-ID or Persistent Tracker)
            retained_id = getattr(track, "retained_id", None) or getattr(track, "persistent_id", None) or f"PER-{tid:04d}"

            if is_flagged:
                color = (0, 0, 230)  # Red alert for flagged face
            elif is_identified:
                cat = str(face_info.get("category", "NORMAL")).upper()
                if cat == "VIP":
                    color = (0, 200, 0)
                elif cat == "STAFF":
                    color = (200, 200, 0)

            # Trajectory trail
            if len(trail) >= 2:
                for idx in range(1, len(trail)):
                    alpha = idx / len(trail)
                    thickness = max(1, int(3 * alpha))
                    pt1 = (int(trail[idx - 1][0]), int(trail[idx - 1][1]))
                    pt2 = (int(trail[idx][0]), int(trail[idx][1]))
                    cv2.line(im, pt1, pt2, color, thickness)

            # Clean bounding box
            box_thick = 3 if is_flagged else 2
            cv2.rectangle(im, (x1, y1), (x2, y2), color, box_thick)

            # Retained ID badge display
            if is_identified:
                name = face_info.get("name", "Identified")
                cat = face_info.get("category", "KNOWN")
                badge_text = f"[{retained_id}] {name} ({cat}) | {speed:.0f} px/s [{level}]"
            else:
                badge_text = f"[{retained_id}] Human | {speed:.0f} px/s [{level}]"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            by1 = max(0, y1 - th - 8)
            cv2.rectangle(im, (x1, by1), (x1 + tw + 8, y1), color, -1)
            text_col = (255, 255, 255) if (is_flagged or level in ("HIGH", "MEDIUM")) else (0, 0, 0)
            cv2.putText(im, badge_text, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_col, 1, cv2.LINE_AA)

        # 2. Render Vehicle Tracks (ANPR Badges with Persistent ID)
        if anpr_results is not None:
            rendered_human_tids = {t.track_id for t in routing_result.human_tracks}
            for track in routing_result.vehicle_tracks:
                tid = track.track_id
                if detector_mode == DetectorMode.SINGLE_CLASS_TEST and tid in rendered_human_tids:
                    # In single class mode (persons), do not draw duplicate vehicle badge
                    continue
                tlwh = track.tlwh
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h

                pid_label = getattr(track, "persistent_id", None) or f"VEH-{tid:04d}"
                plate_info = anpr_results.get(tid)
                if plate_info and plate_info.get("plate_number"):
                    plate = plate_info["plate_number"]
                    conf = plate_info.get("confidence", 0.0)
                    is_flagged = plate_info.get("is_flagged", False)
                    category = plate_info.get("alert_category", "NORMAL").upper()

                    box_color = (0, 0, 230) if is_flagged else (0, 200, 0)
                    box_thick = 3 if is_flagged else 2
                    cv2.rectangle(im, (x1, y1), (x2, y2), box_color, box_thick)

                    tag = f"[{pid_label} | {category}] {plate}" if is_flagged else f"[{pid_label}] {plate}"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    by1 = max(0, y1 - th - 8)
                    cv2.rectangle(im, (x1, by1), (x1 + tw + 8, y1), box_color, -1)
                    cv2.putText(im, tag, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    # Vehicle without plate result yet
                    cv2.rectangle(im, (x1, y1), (x2, y2), (230, 180, 0), 2)
                    tag = f"[{pid_label}] Unidentified"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    by1 = max(0, y1 - th - 8)
                    cv2.rectangle(im, (x1, by1), (x1 + tw + 8, y1), (230, 180, 0), -1)
                    cv2.putText(im, tag, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Top Unified Dashboard Header (optional, disabled in dashboard web streams)
        if show_header:
            hud_h = 45
            overlay = im.copy()
            cv2.rectangle(overlay, (0, 0), (im_w, hud_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.85, im, 0.15, 0, im)

            # Left status text
            mode_tag = "PRODUCTION (2-CLASS)" if detector_mode == DetectorMode.MULTI_CLASS_PRODUCTION else "TEST (SINGLE-CLASS)"
            title_text = f"ROUTER: {mode_tag} | Vehicles: {routing_result.num_vehicles} | Humans: {routing_result.num_humans}"
            cv2.putText(im, title_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 240, 255), 2, cv2.LINE_AA)

            # Right status text
            stats_text = f"Frame: {frame_id} | FPS: {fps:.1f}"
            (stw, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(im, stats_text, (im_w - stw - 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        return im

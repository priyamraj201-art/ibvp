import argparse
import os
import os.path as osp
import sys

# Prevent OpenMP runtime collision on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import cv2
import numpy as np
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.alert_system import MotionAlertSystem
from yolox.tracker.persistent_tracker import GLOBAL_PERSISTENT_TRACKER
from yolox.anpr import ANPRPipeline, ANPRVisualizer
from yolox.frs import FRSPipeline, FRSVisualizer
from yolox.routing import TrackRouter, DetectorMode
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--rotate", default=0, type=int, choices=[0, 90, 180, 270], help="rotate camera feed (0, 90, 180, 270 degrees)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.25, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # Alert system args
    parser.add_argument("--alert", action="store_true", default=True, help="enable motion alert system (LOW, MEDIUM, HIGH)")
    parser.add_argument("--no_alert", dest="alert", action="store_false", help="disable motion alert system")
    parser.add_argument("--alert_low", type=float, default=20.0, help="speed threshold for LOW alert in px/s (default: 20)")
    parser.add_argument("--alert_med", type=float, default=60.0, help="speed threshold for MEDIUM alert in px/s (default: 60)")
    parser.add_argument("--alert_high", type=float, default=120.0, help="speed threshold for HIGH alert in px/s (default: 120)")
    parser.add_argument("--alert_sound", action="store_true", default=False, help="enable audio beep on HIGH motion alert")

    # ANPR args
    parser.add_argument("--anpr", action="store_true", default=False, help="enable ANPR (Automatic Number Plate Recognition)")
    parser.add_argument("--anpr_db", type=str, default="anpr_watchlist.db", help="path to SQLite watchlist database")
    parser.add_argument("--anpr_workers", type=int, default=1, help="number of async OCR worker threads")
    parser.add_argument("--anpr_min_area", type=float, default=800.0, help="min vehicle bbox area to trigger ANPR")
    parser.add_argument("--seed_watchlist", action="store_true", default=False, help="seed database with sample watchlist entries")

    # FRS args
    parser.add_argument("--frs", action="store_true", default=False, help="enable Facial Recognition System")
    parser.add_argument("--frs_db", type=str, default="frs_faces.db", help="path to FRS SQLite face database")
    parser.add_argument("--frs_workers", type=int, default=1, help="number of async FRS worker threads")
    parser.add_argument("--frs_min_area", type=float, default=1500.0, help="min human bbox area to trigger FRS")
    parser.add_argument("--frs_threshold", type=float, default=0.45, help="cosine similarity threshold for face match")
    parser.add_argument("--seed_faces", action="store_true", default=False, help="seed FRS database with sample watchlist identities")
    parser.add_argument("--detect_skip", type=int, default=1, help="run detector every N frames for CPU speedup (e.g. 3 = 3x faster display). Tracking interpolates between detections.")

    # Routing & Multi-class args
    parser.add_argument(
        "--detector_mode",
        default="single_class_test",
        choices=["single_class_test", "multi_class_production"],
        help="Detector routing mode: 'single_class_test' (default MOT17 test) or 'multi_class_production' (class-aware routing)",
    )
    parser.add_argument("--num_classes", type=int, default=None, help="override number of model classes")
    parser.add_argument("--class_names", type=str, default="HUMAN,VEHICLE", help="comma-separated class names for routing")

    # Dual-Model Ensemble args (MOT17 Human SOTA + COCO Vehicles + ANPR)
    parser.add_argument("--ensemble", action="store_true", default=False, help="enable Dual-Model Ensemble: MOT17 model for human SOTA + COCO model for vehicles + ANPR")
    parser.add_argument("--human_exp", type=str, default="exps/example/mot/yolox_x_mix_det.py", help="exp file for human detector")
    parser.add_argument("--human_ckpt", type=str, default="pretrained/bytetrack_x_mot17.pth.tar", help="checkpoint for human detector")
    parser.add_argument("--vehicle_exp", type=str, default="exps/default/yolox_x.py", help="exp file for vehicle detector")
    parser.add_argument("--vehicle_ckpt", type=str, default="pretrained/yolox_x.pth", help="checkpoint for vehicle detector")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


class DualPredictor(object):
    """
    Dual-Model Ensemble Predictor:
    - MOT17/CrowdHuman model for highest-efficiency human detection (Class 0: HUMAN)
    - COCO model (YOLOX-X or YOLOX-S) for full vehicle detection + ANPR (Class 1: VEHICLE)
    """
    def __init__(
        self,
        human_model,
        human_exp,
        vehicle_model,
        vehicle_exp,
        device=torch.device("cpu"),
        fp16=False,
    ):
        self.human_model = human_model
        self.human_exp = human_exp
        self.vehicle_model = vehicle_model
        self.vehicle_exp = vehicle_exp
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.test_size = human_exp.test_size
        self.num_classes = 2

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        with torch.no_grad():
            timer.tic()
            # 1. Human Model Inference (MOT17/CrowdHuman specialized weights)
            img_h, ratio_h = preproc(img, self.human_exp.test_size, self.rgb_means, self.std)
            img_info["ratio"] = ratio_h
            t_h = torch.from_numpy(img_h).unsqueeze(0).float().to(self.device)
            if self.fp16:
                t_h = t_h.half()
            out_h = self.human_model(t_h)
            out_h = postprocess(out_h, self.human_exp.num_classes, self.human_exp.test_conf, self.human_exp.nmsthre)[0]

            if out_h is not None and out_h.shape[0] > 0:
                out_h[:, 6] = 0.0  # Force Class 0: HUMAN

            # 2. Vehicle Model Inference (COCO weights)
            img_v, ratio_v = preproc(img, self.vehicle_exp.test_size, self.rgb_means, self.std)
            t_v = torch.from_numpy(img_v).unsqueeze(0).float().to(self.device)
            if self.fp16:
                t_v = t_v.half()
            out_v = self.vehicle_model(t_v)
            out_v = postprocess(out_v, self.vehicle_exp.num_classes, self.vehicle_exp.test_conf, self.vehicle_exp.nmsthre)[0]

            if out_v is not None and out_v.shape[0] > 0:
                # 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
                veh_mask = torch.isin(out_v[:, 6], torch.tensor([1.0, 2.0, 3.0, 5.0, 7.0], device=self.device))
                out_v = out_v[veh_mask]
                if out_v.shape[0] > 0:
                    if ratio_v != ratio_h:
                        out_v[:, :4] = out_v[:, :4] * (ratio_h / ratio_v)
                    out_v[:, 6] = 1.0  # Force Class 1: VEHICLE
                else:
                    out_v = None

            # 3. Fuse Detections
            if out_h is not None and out_v is not None:
                fused = torch.cat([out_h, out_v], dim=0)
            elif out_h is not None:
                fused = out_h
            elif out_v is not None:
                fused = out_v
            else:
                fused = None

        return [fused], img_info


def image_demo(predictor, vis_folder, current_time, args, exp):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    if exp.num_classes == 80:
        class_names = TrackRouter.COCO_CLASSES
    else:
        class_names = args.class_names.split(",") if isinstance(args.class_names, str) else args.class_names
    track_router = TrackRouter(class_names=class_names, default_mode=args.detector_mode)

    anpr_pipeline = ANPRPipeline(
        db_path=args.anpr_db,
        min_box_area=args.anpr_min_area,
        num_workers=args.anpr_workers,
    ) if args.anpr else None

    if anpr_pipeline and args.seed_watchlist:
        anpr_pipeline.watchlist_db.seed_sample_watchlist()

    frs_pipeline = FRSPipeline(
        db_path=args.frs_db,
        min_box_area=args.frs_min_area,
        num_workers=args.frs_workers,
        match_threshold=args.frs_threshold,
    ) if args.frs else None

    if frs_pipeline and args.seed_faces:
        frs_pipeline.face_db.seed_sample_identities()

    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            valid_targets = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = False
                if args.detector_mode == "single_class_test" and args.aspect_ratio_thresh > 0:
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    valid_targets.append(t)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()

            routing_result = track_router.route(valid_targets, detector_mode=args.detector_mode)
            anpr_results = None
            frs_results = None

            if anpr_pipeline is not None:
                veh_tlwhs = [t.tlwh for t in routing_result.vehicle_tracks]
                veh_ids = [t.track_id for t in routing_result.vehicle_tracks]
                veh_scores = [t.score for t in routing_result.vehicle_tracks]
                anpr_results = anpr_pipeline.process_frame(
                    img_info['raw_img'], veh_tlwhs, veh_ids, veh_scores, current_time=time.time()
                )

            if frs_pipeline is not None:
                hum_tlwhs = [t.tlwh for t in routing_result.human_tracks]
                hum_ids = [t.track_id for t in routing_result.human_tracks]
                hum_scores = [t.score for t in routing_result.human_tracks]
                frs_results = frs_pipeline.process_frame(
                    img_info['raw_img'], hum_tlwhs, hum_ids, hum_scores, current_time=time.time()
                )

            if args.detector_mode == DetectorMode.MULTI_CLASS_PRODUCTION:
                online_im = track_router.draw_unified_overlay(
                    img_info['raw_img'], routing_result, anpr_results=anpr_results, alert_data=None, frs_results=frs_results,
                    detector_mode=args.detector_mode, frame_id=frame_id, fps=1. / max(1e-5, timer.average_time)
                )
            elif frs_pipeline is not None and anpr_pipeline is None:
                all_tlwhs = [t.tlwh for t in valid_targets]
                all_ids = [t.track_id for t in valid_targets]
                online_im = FRSVisualizer.draw_frs_overlay(
                    img_info['raw_img'], all_tlwhs, all_ids, frs_results, frame_id=frame_id, fps=1. / max(1e-5, timer.average_time)
                )
            elif anpr_pipeline is not None:
                all_tlwhs = [t.tlwh for t in valid_targets]
                all_ids = [t.track_id for t in valid_targets]
                online_im = ANPRVisualizer.draw_anpr_overlay(
                    img_info['raw_img'], all_tlwhs, all_ids, anpr_results, frame_id=frame_id, fps=1. / max(1e-5, timer.average_time)
                )
            else:
                all_tlwhs = [t.tlwh for t in valid_targets]
                all_ids = [t.track_id for t in valid_targets]
                online_im = plot_tracking(
                    img_info['raw_img'], all_tlwhs, all_ids, frame_id=frame_id, fps=1. / max(1e-5, timer.average_time)
                )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        cv2.imshow("ByteTrack", online_im)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if anpr_pipeline is not None:
        anpr_pipeline.stop()
    if frs_pipeline is not None:
        frs_pipeline.stop()
    cv2.destroyAllWindows()
    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def imageflow_demo(predictor, vis_folder, current_time, args, exp):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    if not cap.isOpened():
        logger.error(f"Could not open {'video ' + args.path if args.demo == 'video' else 'webcam (camid=' + str(args.camid) + ')'}!")
        return
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    if args.rotate in (90, 270):
        width, height = height, width
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    if exp.num_classes == 80:
        class_names = TrackRouter.COCO_CLASSES
    else:
        class_names = args.class_names.split(",") if isinstance(args.class_names, str) else args.class_names
    track_router = TrackRouter(class_names=class_names, default_mode=args.detector_mode)

    alert_system = MotionAlertSystem(
        low_thresh=args.alert_low,
        med_thresh=args.alert_med,
        high_thresh=args.alert_high,
        enable_sound=args.alert_sound,
    ) if (args.alert and (args.detector_mode == "multi_class_production" or not (args.anpr or args.frs))) else None

    anpr_pipeline = ANPRPipeline(
        db_path=args.anpr_db,
        min_box_area=args.anpr_min_area,
        num_workers=args.anpr_workers,
    ) if args.anpr else None

    if anpr_pipeline and args.seed_watchlist:
        anpr_pipeline.watchlist_db.seed_sample_watchlist()

    frs_pipeline = FRSPipeline(
        db_path=args.frs_db,
        min_box_area=args.frs_min_area,
        num_workers=args.frs_workers,
        match_threshold=args.frs_threshold,
    ) if args.frs else None

    if frs_pipeline and args.seed_faces:
        frs_pipeline.face_db.seed_sample_identities()

    timer = Timer()
    frame_id = 0
    results = []
    detect_skip = max(1, getattr(args, 'detect_skip', 1))
    # State carried across skipped frames
    _last_outputs = None
    _last_img_info = None
    _last_valid_targets = []
    _last_routing_result = None
    _last_frs_results = None
    _last_anpr_results = None
    _last_alert_data = None
    _last_online_im = None

    def _enroll_face_interactive(raw_frame, valid_targets, frs_pipeline):
        """Pause and enroll the largest detected face into FRS database interactively."""
        if frs_pipeline is None:
            logger.warning("FRS not enabled. Run with --frs to enable face enrollment.")
            return
        # Find largest bounding box among valid targets
        best_t = None
        best_area = 0
        for t in valid_targets:
            area = t.tlwh[2] * t.tlwh[3]
            if area > best_area:
                best_area = area
                best_t = t
        if best_t is None:
            logger.warning("No tracked person in frame to enroll. Move closer to camera.")
            return
        # Crop the person region
        x, y, w, h = [int(v) for v in best_t.tlwh]
        x2, y2 = min(x + w, raw_frame.shape[1]), min(y + h, raw_frame.shape[0])
        x = max(0, x); y = max(0, y)
        crop = raw_frame[y:y2, x:x2]
        if crop.size == 0:
            logger.warning("Empty crop, could not enroll.")
            return
        # Show enrollment preview in a separate window
        preview = crop.copy()
        cv2.putText(preview, "ENROLLING THIS FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("FRS Enrollment Preview", preview)
        cv2.waitKey(500)
        cv2.destroyWindow("FRS Enrollment Preview")
        # Console prompt (non-blocking - printed to terminal)
        print("\n" + "="*60)
        print("  FRS LIVE ENROLLMENT")
        print("="*60)
        print("  Track ID  :", best_t.track_id)
        print("  Bbox area :", int(best_area), "px²")
        print("-"*60)
        person_id = input("  Enter Person ID (e.g. STAFF_001) [blank=cancel]: ").strip()
        if not person_id:
            print("  [Cancelled]")
            print("="*60)
            return
        full_name = input("  Enter Full Name (e.g. Arjun Sharma): ").strip() or person_id
        category = input("  Category [STAFF/VIP/SUSPECT/WANTED/UNKNOWN] (default STAFF): ").strip().upper() or "STAFF"
        notes = input("  Notes (optional): ").strip()
        enrolled_by = "live_webcam"
        # Extract embedding via FRS pipeline embedder (ArcFace)
        detector = frs_pipeline.detector
        embedder = frs_pipeline.embedder
        best_face = detector.get_best_face(crop)
        if best_face is None:
            logger.warning("No face detected in the person crop. Try again with a clearer view of the face.")
            print("  [FAILED] No face detected in crop.")
            print("="*60)
            return
        face_crop, score = best_face
        emb, quality = embedder.get_embedding(face_crop)
        if emb is None:
            logger.warning("Could not extract face embedding.")
            print("  [FAILED] Could not extract embedding.")
            print("="*60)
            return
        obs_id = frs_pipeline.face_db.enroll_face(
            person_id=person_id,
            name=full_name,
            embedding=emb,
            category=category,
            enrolled_by=enrolled_by,
            notes=notes,
        )
        if obs_id:
            print(f"  [SUCCESS] Enrolled '{full_name}' ({person_id}) as {category}")
            logger.info(f"Live enrolled: {full_name} ({person_id}) category={category}")
        else:
            print("  [FAILED] Enrollment failed. Check logs.")
        print("="*60)

    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            if args.rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif args.rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif args.rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # --- Frame-skip: only run detector every detect_skip frames ---
            run_detection = (frame_id % detect_skip == 0)

            if run_detection:
                timer.tic()
                outputs, img_info = predictor.inference(frame, timer)
                _last_outputs = outputs
                _last_img_info = img_info
            else:
                # Reuse last detection result with fresh raw frame for display
                outputs = _last_outputs
                if _last_img_info is not None:
                    img_info = dict(_last_img_info)
                    img_info['raw_img'] = frame
                else:
                    img_info = {'raw_img': frame, 'height': frame.shape[0], 'width': frame.shape[1]}

            if outputs is not None and outputs[0] is not None:
                if run_detection:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                    valid_targets = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = False
                        if args.detector_mode == "single_class_test" and args.aspect_ratio_thresh > 0:
                            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            valid_targets.append(t)
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                    _last_valid_targets = valid_targets
                    timer.toc()
                else:
                    valid_targets = _last_valid_targets

                # Class-Aware Routing
                routing_result = track_router.route(valid_targets, detector_mode=args.detector_mode)
                anpr_results = None
                alert_data = None
                frs_results = None

                if alert_system is not None:
                    hum_tlwhs = [t.tlwh for t in routing_result.human_tracks]
                    hum_ids = [t.track_id for t in routing_result.human_tracks]
                    alert_data = alert_system.update(hum_tlwhs, hum_ids, current_time=time.time())

                # Persistent ID allocation for every tracked target
                human_ids_set = {t.track_id for t in routing_result.human_tracks}
                vehicle_ids_set = {t.track_id for t in routing_result.vehicle_tracks}
                for t in valid_targets:
                    etype = "HUMAN" if t.track_id in human_ids_set else ("VEHICLE" if t.track_id in vehicle_ids_set else "OBJECT")
                    spd = alert_data.get(t.track_id, {}).get("speed", 0.0) if alert_data else 0.0
                    raw_f = img_info['raw_img'] if isinstance(img_info, dict) and 'raw_img' in img_info else frame
                    t.persistent_id = GLOBAL_PERSISTENT_TRACKER.assign_or_get_id(
                        cam_id=getattr(args, "camid", 0),
                        cam_name="CLI Video Tracker",
                        local_track_id=t.track_id,
                        entity_type=etype,
                        tlwh=t.tlwh,
                        frame=raw_f,
                        speed=spd,
                    )

                if anpr_pipeline is not None:
                    veh_tlwhs = [t.tlwh for t in routing_result.vehicle_tracks]
                    veh_ids = [t.track_id for t in routing_result.vehicle_tracks]
                    veh_scores = [t.score for t in routing_result.vehicle_tracks]
                    anpr_results = anpr_pipeline.process_frame(
                        img_info['raw_img'], veh_tlwhs, veh_ids, veh_scores, current_time=time.time()
                    )
                    if anpr_results:
                        for tid, res in anpr_results.items():
                            plate = res.get("plate_number")
                            is_id = bool(plate)
                            GLOBAL_PERSISTENT_TRACKER.update_identification(
                                cam_id=getattr(args, "camid", 0),
                                local_track_id=tid,
                                is_identified=is_id,
                                identified_id=plate or "UNIDENTIFIED",
                                identified_name=f"Vehicle {plate}" if plate else "Unidentified Vehicle",
                                category=res.get("alert_category", "NORMAL"),
                                confidence=res.get("confidence", 0.0),
                                cam_name="CLI Video Tracker",
                            )

                if frs_pipeline is not None:
                    hum_tlwhs = [t.tlwh for t in routing_result.human_tracks]
                    hum_ids = [t.track_id for t in routing_result.human_tracks]
                    hum_scores = [t.score for t in routing_result.human_tracks]
                    frs_results = frs_pipeline.process_frame(
                        img_info['raw_img'], hum_tlwhs, hum_ids, hum_scores, current_time=time.time()
                    )
                    if frs_results:
                        for tid, res in frs_results.items():
                            is_id = bool(res.get("person_id") and res.get("person_id") != "UNKNOWN")
                            GLOBAL_PERSISTENT_TRACKER.update_identification(
                                cam_id=getattr(args, "camid", 0),
                                local_track_id=tid,
                                is_identified=is_id,
                                identified_id=res.get("person_id", "UNIDENTIFIED"),
                                identified_name=res.get("name", "Unidentified Person"),
                                category=res.get("category", "NORMAL"),
                                confidence=res.get("confidence", 0.0),
                                cam_name="CLI Video Tracker",
                            )
                _last_routing_result = routing_result
                _last_frs_results = frs_results
                _last_anpr_results = anpr_results
                _last_alert_data = alert_data

                # Render unified or specialized output
                if args.detector_mode == DetectorMode.MULTI_CLASS_PRODUCTION:
                    online_im = track_router.draw_unified_overlay(
                        img_info['raw_img'], routing_result, anpr_results=anpr_results, alert_data=alert_data, frs_results=frs_results,
                        detector_mode=args.detector_mode, frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
                    )
                elif frs_pipeline is not None and anpr_pipeline is None:
                    all_tlwhs = [t.tlwh for t in valid_targets]
                    all_ids = [t.track_id for t in valid_targets]
                    online_im = FRSVisualizer.draw_frs_overlay(
                        img_info['raw_img'], all_tlwhs, all_ids, frs_results, frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
                    )
                elif anpr_pipeline is not None:
                    all_tlwhs = [t.tlwh for t in valid_targets]
                    all_ids = [t.track_id for t in valid_targets]
                    online_im = ANPRVisualizer.draw_anpr_overlay(
                        img_info['raw_img'], all_tlwhs, all_ids, anpr_results, frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
                    )
                elif alert_system is not None:
                    all_tlwhs = [t.tlwh for t in valid_targets]
                    all_ids = [t.track_id for t in valid_targets]
                    online_im = alert_system.draw_alerts(
                        img_info['raw_img'], all_tlwhs, all_ids, alert_data, frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
                    )
                else:
                    all_tlwhs = [t.tlwh for t in valid_targets]
                    all_ids = [t.track_id for t in valid_targets]
                    online_im = plot_tracking(
                        img_info['raw_img'], all_tlwhs, all_ids, frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
                    )
            else:
                if run_detection:
                    timer.toc()
                routing_result = track_router.route([], detector_mode=args.detector_mode)
                anpr_results = None
                alert_data = None
                frs_results = None
                if anpr_pipeline is not None:
                    anpr_results = anpr_pipeline.process_frame(
                        img_info['raw_img'], [], [], [], current_time=time.time()
                    )
                if alert_system is not None:
                    alert_data = alert_system.update([], [], current_time=time.time())
                if frs_pipeline is not None:
                    frs_results = frs_pipeline.process_frame(
                        img_info['raw_img'], [], [], [], current_time=time.time()
                    )

                if args.detector_mode == DetectorMode.MULTI_CLASS_PRODUCTION:
                    online_im = track_router.draw_unified_overlay(
                        img_info['raw_img'], routing_result, anpr_results=anpr_results, alert_data=alert_data, frs_results=frs_results,
                        detector_mode=args.detector_mode, frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
                    )
                elif frs_pipeline is not None and anpr_pipeline is None:
                    online_im = FRSVisualizer.draw_frs_overlay(
                        img_info['raw_img'], [], [], frs_results, frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
                    )
                elif anpr_pipeline is not None:
                    online_im = ANPRVisualizer.draw_anpr_overlay(
                        img_info['raw_img'], [], [], anpr_results, frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
                    )
                elif alert_system is not None:
                    online_im = alert_system.draw_alerts(
                        img_info['raw_img'], [], [], alert_data, frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
                    )
                else:
                    online_im = img_info['raw_img']

            # Overlay enrollment hint if FRS enabled (webcam only)
            if frs_pipeline is not None and args.demo in ("webcam",):
                hint = "Press [E] to enroll face | [Q] to quit"
                cv2.putText(online_im, hint, (10, online_im.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)

            _last_online_im = online_im
            if args.save_result:
                vid_writer.write(online_im)
            cv2.imshow("ByteTrack", online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            elif ch == ord("e") or ch == ord("E"):
                # Live face enrollment - pause display, prompt in terminal
                if _last_img_info is not None:
                    _raw = frame.copy()
                    _enroll_face_interactive(_raw, _last_valid_targets, frs_pipeline)
        else:
            break
        frame_id += 1

    if anpr_pipeline is not None:
        anpr_pipeline.stop()
    if frs_pipeline is not None:
        frs_pipeline.stop()
    cap.release()
    if args.save_result:
        vid_writer.release()
    cv2.destroyAllWindows()

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    if args.save_result:
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    # Auto-detect CUDA availability
    if args.device == "gpu":
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
        else:
            logger.warning("CUDA is not available or torch was not compiled with CUDA. Falling back to CPU.")
            args.device = torch.device("cpu")
            args.fp16 = False
    else:
        args.device = torch.device("cpu")
        args.fp16 = False

    logger.info("Args: {}".format(args))

    if args.ensemble:
        logger.info("==================================================================")
        logger.info("INITIALIZING DUAL-MODEL ENSEMBLE:")
        logger.info(f" -> Human Detector (SOTA MOT17): {args.human_ckpt} ({args.human_exp})")
        logger.info(f" -> Vehicle Detector (COCO):     {args.vehicle_ckpt} ({args.vehicle_exp})")
        logger.info("==================================================================")

        human_exp = get_exp(args.human_exp, None)
        if args.conf is not None:
            human_exp.test_conf = args.conf
        human_model = human_exp.get_model().to(args.device)
        ckpt_h = torch.load(args.human_ckpt, map_location="cpu")
        human_model.load_state_dict(ckpt_h["model"] if "model" in ckpt_h else ckpt_h)
        human_model.eval()

        vehicle_exp = get_exp(args.vehicle_exp, None)
        vehicle_exp.test_conf = 0.1 if args.conf is None else args.conf
        vehicle_model = vehicle_exp.get_model().to(args.device)
        ckpt_v = torch.load(args.vehicle_ckpt, map_location="cpu")
        vehicle_model.load_state_dict(ckpt_v["model"] if "model" in ckpt_v else ckpt_v)
        vehicle_model.eval()

        if args.fuse:
            human_model = fuse_model(human_model)
            vehicle_model = fuse_model(vehicle_model)
        if args.fp16:
            human_model = human_model.half()
            vehicle_model = vehicle_model.half()

        predictor = DualPredictor(human_model, human_exp, vehicle_model, vehicle_exp, args.device, args.fp16)
        current_time = time.localtime()
        exp.test_size = human_exp.test_size
        exp.num_classes = 2
        args.detector_mode = "multi_class_production"
        args.class_names = "HUMAN,VEHICLE"
        if args.demo == "image":
            image_demo(predictor, vis_folder, current_time, args, exp)
        elif args.demo == "video" or args.demo == "webcam":
            imageflow_demo(predictor, vis_folder, current_time, args, exp)
        return

    if args.num_classes is not None:
        exp.num_classes = args.num_classes
    if args.conf is not None:
        exp.test_conf = args.conf
    elif exp.test_conf < 0.01:
        exp.test_conf = 0.1  # Default production confidence threshold
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info(f"loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        ckpt_state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model_state_dict = model.state_dict()

        # Adapt class prediction layers if checkpoint num_classes differs from model num_classes
        for k in list(ckpt_state_dict.keys()):
            if k in model_state_dict:
                v_ckpt = ckpt_state_dict[k]
                v_model = model_state_dict[k]
                if v_ckpt.shape != v_model.shape:
                    logger.warning(
                        f"Adapting weight layer '{k}' from checkpoint shape {list(v_ckpt.shape)} to model shape {list(v_model.shape)}"
                    )
                    adapted = v_model.clone()
                    if len(v_ckpt.shape) == 4 and v_ckpt.shape[1:] == v_model.shape[1:]:
                        if v_ckpt.shape[0] == 80 and v_model.shape[0] == 2:
                            # COCO 80-class mapping:
                            # 0: person -> HUMAN (class 0)
                            # 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck -> VEHICLE (class 1)
                            adapted[0] = v_ckpt[0]
                            adapted[1] = v_ckpt[[1, 2, 3, 5, 7]].mean(dim=0)
                            logger.info(f"Mapped COCO 80-class layer '{k}' to [HUMAN, VEHICLE]")
                        elif v_ckpt.shape[0] == 1 and v_model.shape[0] == 2:
                            adapted[0] = v_ckpt[0]          # Class 0: HUMAN gets full MOT17 pedestrian detector weights
                            adapted[1] = v_ckpt[0] * 0.01   # Class 1: VEHICLE lower baseline until fine-tuned
                        else:
                            min_classes = min(v_ckpt.shape[0], v_model.shape[0])
                            adapted[:min_classes] = v_ckpt[:min_classes]
                        ckpt_state_dict[k] = adapted
                    elif len(v_ckpt.shape) == 1:
                        if v_ckpt.shape[0] == 80 and v_model.shape[0] == 2:
                            adapted[0] = v_ckpt[0]
                            adapted[1] = v_ckpt[[1, 2, 3, 5, 7]].mean(dim=0)
                        elif v_ckpt.shape[0] == 1 and v_model.shape[0] == 2:
                            adapted[0] = v_ckpt[0]
                            adapted[1] = v_ckpt[0] - 3.0    # Class 1: lower logit bias
                        else:
                            min_classes = min(v_ckpt.shape[0], v_model.shape[0])
                            adapted[:min_classes] = v_ckpt[:min_classes]
                        ckpt_state_dict[k] = adapted
                    else:
                        del ckpt_state_dict[k]

        model.load_state_dict(ckpt_state_dict, strict=False)
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args, exp)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args, exp)


if __name__ == "__main__":
    args = make_parser().parse_args()
    if args.ensemble and args.exp_file is None:
        args.exp_file = args.human_exp
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)

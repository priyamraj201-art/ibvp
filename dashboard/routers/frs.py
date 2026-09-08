"""
FRS router — identity management, recognition logs, live face enrollment.
"""

import os
import sys
import tempfile
import base64
from typing import Optional

from fastapi import APIRouter, Request, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np

from dashboard import db
from dashboard.stream_server import STREAM_STATE

router = APIRouter()

_templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_templates_dir)

_bytetrack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)


def _get_frs_db(request: Request) -> str:
    return request.app.state.frs_db


def _enroll_face_image(
    img: np.ndarray,
    person_id: str,
    name: str,
    category: str,
    notes: str,
    enrolled_by: str,
    frs_db: str,
) -> tuple[bool, str]:
    """Extract face, compute ArcFace 512-D embedding, and register in SQLite database."""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from yolox.frs.face_detector import FaceDetector
    from yolox.frs.face_embedder import FaceEmbedder
    from yolox.frs.face_database import FaceDatabase

    detector = FaceDetector()
    embedder = FaceEmbedder()

    face_info = detector.get_best_face(img)
    if face_info is None:
        return False, "No face detected in the image. Please position the face centered, looking toward the camera."

    face_crop, det_conf = face_info
    embedding, quality = embedder.get_embedding(face_crop)

    if quality < 0.05:
        return False, "Face clarity is too low or blurry. Please capture a sharper image."

    fdb = FaceDatabase(db_path=frs_db)
    ok = fdb.enroll_face(
        person_id=person_id.strip().upper(),
        name=name.strip() or person_id.strip().upper(),
        embedding=embedding,
        category=category.strip().upper(),
        notes=notes.strip(),
        enrolled_by=enrolled_by.strip() or "admin",
    )

    if ok:
        return True, f"Successfully enrolled {person_id.strip().upper()} ({name or person_id}) as {category.strip().upper()}."
    return False, "Database insertion failed."


@router.get("/frs/identities")
async def frs_identities_page(request: Request):
    frs_db = _get_frs_db(request)
    identities = db.get_frs_identities(frs_db)
    return templates.TemplateResponse(request=request, name="frs_identities.html", context={
        "request": request,
        "identities": identities,
    })


@router.get("/frs/logs")
async def frs_logs_page(request: Request, category: str = None,
                         date_from: str = None, date_to: str = None):
    frs_db = _get_frs_db(request)
    logs = db.get_frs_logs(frs_db, category=category, date_from=date_from, date_to=date_to)
    return templates.TemplateResponse(request=request, name="frs_logs.html", context={
        "request": request,
        "logs": logs,
        "filter_category": category or "",
        "filter_date_from": date_from or "",
        "filter_date_to": date_to or "",
    })


@router.get("/frs/enroll")
async def frs_enroll_page(request: Request):
    return templates.TemplateResponse("frs_enroll.html", {"request": request})


@router.post("/api/frs/enroll")
async def frs_enroll_api(
    request: Request,
    person_id: str = Form(...),
    name: str = Form(""),
    category: str = Form("STAFF"),
    notes: str = Form(""),
    enrolled_by: str = Form("admin"),
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
):
    frs_db = _get_frs_db(request)
    img = None

    try:
        # 1. Base64 payload from browser live webcam
        if image_base64:
            clean_b64 = image_base64
            if "," in clean_b64:
                clean_b64 = clean_b64.split(",", 1)[1]
            raw_data = base64.b64decode(clean_b64)
            nparr = np.frombuffer(raw_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Multipart file upload
        elif image is not None:
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return JSONResponse({"success": False, "message": "No valid image data provided."}, status_code=400)

        ok, msg = _enroll_face_image(img, person_id, name, category, notes, enrolled_by, frs_db)
        return JSONResponse({"success": ok, "message": msg}, status_code=200 if ok else 400)

    except Exception as e:
        return JSONResponse({"success": False, "message": f"Error: {str(e)}"}, status_code=500)


@router.post("/api/frs/enroll-live-stream")
async def frs_enroll_live_stream(
    request: Request,
    body: dict = Body(...),
):
    """Enrolls the face currently visible in the active surveillance stream."""
    frs_db = _get_frs_db(request)
    frame_bytes = STREAM_STATE.get_frame()

    if frame_bytes is None:
        return JSONResponse({"success": False, "message": "Live stream is not running. Please start the stream first."}, status_code=400)

    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"success": False, "message": "Failed to decode current live frame."}, status_code=500)

    person_id = body.get("person_id", "").strip()
    name = body.get("name", "").strip()
    category = body.get("category", "STAFF").strip()
    notes = body.get("notes", "Enrolled directly from live surveillance feed").strip()
    enrolled_by = body.get("enrolled_by", "operator").strip()

    if not person_id:
        return JSONResponse({"success": False, "message": "Person ID is required."}, status_code=400)

    ok, msg = _enroll_face_image(img, person_id, name, category, notes, enrolled_by, frs_db)
    return JSONResponse({"success": ok, "message": msg}, status_code=200 if ok else 400)


@router.delete("/api/frs/identity/{person_id}")
async def delete_frs_identity(request: Request, person_id: str):
    frs_db = _get_frs_db(request)
    ok = db.delete_frs_identity(frs_db, person_id)
    if ok:
        return JSONResponse({"success": True})
    return JSONResponse({"success": False, "message": "Delete failed."}, status_code=500)


@router.post("/api/frs/seed")
async def seed_frs_api(request: Request):
    frs_db = _get_frs_db(request)
    ok = db.seed_frs(frs_db)
    if ok:
        return JSONResponse({"success": True, "message": "Sample FRS identities seeded."})
    return JSONResponse({"success": False, "message": "Seeding failed."}, status_code=500)

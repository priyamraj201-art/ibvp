"""
ANPR router — watchlist management, detection logs, add plate.
"""

import os
from fastapi import APIRouter, Request, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from dashboard import db

router = APIRouter()

_templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_templates_dir)


def _get_anpr_db(request: Request) -> str:
    return request.app.state.anpr_db


@router.get("/anpr/watchlist")
async def anpr_watchlist_page(request: Request):
    anpr_db = _get_anpr_db(request)
    watchlist = db.get_anpr_watchlist(anpr_db)
    return templates.TemplateResponse(request=request, name="anpr_watchlist.html", context={
        "request": request,
        "watchlist": watchlist,
    })


@router.get("/anpr/logs")
async def anpr_logs_page(request: Request):
    anpr_db = _get_anpr_db(request)
    logs = db.get_anpr_logs(anpr_db)
    return templates.TemplateResponse(request=request, name="anpr_logs.html", context={
        "request": request,
        "logs": logs,
    })


@router.get("/anpr/add")
async def anpr_add_page(request: Request):
    return templates.TemplateResponse(request=request, name="anpr_add.html", context={"request": request})


@router.post("/api/anpr/plate")
async def add_plate_api(
    request: Request,
    plate_number: str = Form(...),
    owner_name: str = Form(""),
    alert_category: str = Form("SUSPICIOUS"),
    notes: str = Form(""),
):
    anpr_db = _get_anpr_db(request)
    ok = db.add_anpr_plate(anpr_db, plate_number, owner_name, alert_category, notes)
    if ok:
        return JSONResponse({"success": True, "message": f"Plate {plate_number.upper()} added."})
    return JSONResponse({"success": False, "message": "Failed to add plate."}, status_code=500)


@router.delete("/api/anpr/plate/{plate_number}")
async def delete_plate_api(request: Request, plate_number: str):
    anpr_db = _get_anpr_db(request)
    ok = db.delete_anpr_plate(anpr_db, plate_number)
    if ok:
        return JSONResponse({"success": True})
    return JSONResponse({"success": False, "message": "Delete failed."}, status_code=500)


@router.post("/api/anpr/seed")
async def seed_anpr_api(request: Request):
    anpr_db = _get_anpr_db(request)
    ok = db.seed_anpr(anpr_db)
    if ok:
        return JSONResponse({"success": True, "message": "Sample ANPR watchlist seeded."})
    return JSONResponse({"success": False, "message": "Seeding failed."}, status_code=500)

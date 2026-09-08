"""
Settings page router — DB paths, module status, seed buttons, CLI commands.
"""

import os
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()

_templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_templates_dir)

_bytetrack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@router.get("/settings")
async def settings_page(request: Request):
    frs_db = request.app.state.frs_db
    anpr_db = request.app.state.anpr_db

    # Resolve absolute paths
    frs_db_abs = os.path.abspath(frs_db)
    anpr_db_abs = os.path.abspath(anpr_db)

    # Module status checks
    checks = {
        "frs_faces.db": os.path.isfile(frs_db_abs),
        "anpr_watchlist.db": os.path.isfile(anpr_db_abs),
        "Human Model (bytetrack_x_mot17.pth.tar)": os.path.isfile(
            os.path.join(_bytetrack_root, "pretrained", "bytetrack_x_mot17.pth.tar")
        ),
        "Vehicle Model (yolox_x.pth)": os.path.isfile(
            os.path.join(_bytetrack_root, "pretrained", "yolox_x.pth")
        ),
        "Human Exp File": os.path.isfile(
            os.path.join(_bytetrack_root, "exps", "example", "mot", "yolox_x_mix_det.py")
        ),
        "Vehicle Exp File": os.path.isfile(
            os.path.join(_bytetrack_root, "exps", "default", "yolox_x.py")
        ),
    }

    return templates.TemplateResponse(request=request, name="settings.html", context={
        "request": request,
        "frs_db_path": frs_db_abs,
        "anpr_db_path": anpr_db_abs,
        "checks": checks,
    })

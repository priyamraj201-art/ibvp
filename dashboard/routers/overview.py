"""
Overview page router — dashboard home with stat cards.
"""

import os
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from dashboard import db
from yolox.tracker.persistent_tracker import GLOBAL_PERSISTENT_TRACKER

router = APIRouter()

_templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_templates_dir)


def _get_db_paths(request: Request):
    return request.app.state.frs_db, request.app.state.anpr_db


@router.get("/")
async def overview_page(request: Request):
    frs_db, anpr_db = _get_db_paths(request)
    ent_stats = GLOBAL_PERSISTENT_TRACKER.db.get_stats()
    ctx = {
        "request": request,
        "frs_count": db.get_frs_identity_count(frs_db),
        "anpr_count": db.get_anpr_watchlist_count(anpr_db),
        "frs_alerts_today": db.get_frs_alerts_today(frs_db),
        "anpr_alerts_today": db.get_anpr_alerts_today(anpr_db),
        "last_frs": db.get_last_frs_match(frs_db),
        "last_anpr": db.get_last_anpr_match(anpr_db),
        "recent_frs_flagged": db.get_recent_frs_flagged(frs_db),
        "recent_anpr_flagged": db.get_recent_anpr_flagged(anpr_db),
        "entity_stats": ent_stats,
    }
    return templates.TemplateResponse(request=request, name="overview.html", context=ctx)


@router.get("/api/stats")
async def stats_api(request: Request):
    frs_db, anpr_db = _get_db_paths(request)
    return JSONResponse({
        "frs_count": db.get_frs_identity_count(frs_db),
        "anpr_count": db.get_anpr_watchlist_count(anpr_db),
        "frs_alerts_today": db.get_frs_alerts_today(frs_db),
        "anpr_alerts_today": db.get_anpr_alerts_today(anpr_db),
        "last_frs": db.get_last_frs_match(frs_db),
        "last_anpr": db.get_last_anpr_match(anpr_db),
    })

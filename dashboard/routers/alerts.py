"""
Unified alerts router — merged FRS + ANPR flagged logs.
"""

import os
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from dashboard import db

router = APIRouter()

_templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_templates_dir)


@router.get("/alerts")
async def alerts_page(request: Request, alert_type: str = "all"):
    frs_db = request.app.state.frs_db
    anpr_db = request.app.state.anpr_db
    alerts = db.get_unified_alerts(frs_db, anpr_db, alert_type=alert_type)
    return templates.TemplateResponse(request=request, name="alerts.html", context={
        "request": request,
        "alerts": alerts,
        "filter_type": alert_type,
    })

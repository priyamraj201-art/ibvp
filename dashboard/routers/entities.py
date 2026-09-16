"""
Entities Router — Persistent Tracked Entities Management & Cross-Camera Audit Registry.
"""

import os
from typing import Optional
from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from yolox.tracker.persistent_tracker import GLOBAL_PERSISTENT_TRACKER

router = APIRouter()

_templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_templates_dir)


@router.get("/entities")
async def entities_page(request: Request):
    """Render the Persistent Entities command registry page."""
    stats = GLOBAL_PERSISTENT_TRACKER.db.get_stats()
    entities = GLOBAL_PERSISTENT_TRACKER.db.get_entities(limit=100)
    ctx = {
        "request": request,
        "active_page": "entities",
        "stats": stats,
        "entities": entities,
    }
    return templates.TemplateResponse(request=request, name="entities.html", context=ctx)


@router.get("/api/entities")
async def api_get_entities(
    entity_type: Optional[str] = Query(None),
    is_identified: Optional[bool] = Query(None),
    category: Optional[str] = Query(None),
    camera: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Query persistent entities with filtering and search."""
    entities = GLOBAL_PERSISTENT_TRACKER.db.get_entities(
        entity_type=entity_type,
        is_identified=is_identified,
        category=category,
        camera=camera,
        query=query,
        limit=limit,
        offset=offset,
    )
    stats = GLOBAL_PERSISTENT_TRACKER.db.get_stats()
    return JSONResponse({
        "status": "success",
        "count": len(entities),
        "entities": entities,
        "stats": stats,
    })


@router.get("/api/entities/stats")
async def api_entities_stats():
    """Retrieve aggregate telemetry counts for tracked entities."""
    return JSONResponse(GLOBAL_PERSISTENT_TRACKER.db.get_stats())


@router.get("/api/entities/{persistent_id}")
async def api_get_entity_detail(persistent_id: str):
    """Retrieve full details and chronological sighting history for a single entity."""
    entity = GLOBAL_PERSISTENT_TRACKER.db.get_entity(persistent_id)
    if not entity:
        return JSONResponse({"status": "error", "message": "Entity not found"}, status_code=404)
    sightings = GLOBAL_PERSISTENT_TRACKER.db.get_sightings(persistent_id, limit=150)
    return JSONResponse({
        "status": "success",
        "entity": entity,
        "sightings": sightings,
    })

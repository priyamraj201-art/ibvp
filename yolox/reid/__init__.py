"""
Full-Body Person Re-Identification (Person Re-ID) Module for IBVAP.
Provides OSNet feature extraction, universal Retained ID gallery matching,
and asynchronous cross-camera tracking.
"""

from .reid_extractor import PersonReIDExtractor
from .suspect_registry import (
    PersonReIDRegistry,
    RetainedPerson,
    GLOBAL_REID_REGISTRY,
    SuspectRegistry,
    IntrusionDossier,
    GLOBAL_SUSPECT_REGISTRY,
)
from .reid_pipeline import ReIDPipeline, GLOBAL_REID_PIPELINE

__all__ = [
    "PersonReIDExtractor",
    "PersonReIDRegistry",
    "RetainedPerson",
    "GLOBAL_REID_REGISTRY",
    "SuspectRegistry",
    "IntrusionDossier",
    "GLOBAL_SUSPECT_REGISTRY",
    "ReIDPipeline",
    "GLOBAL_REID_PIPELINE",
]

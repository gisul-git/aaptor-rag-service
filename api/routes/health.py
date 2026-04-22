from __future__ import annotations

from fastapi import APIRouter
from core import state

router = APIRouter(tags=["health"])


@router.get("/api/v1/health")
async def health():
    stats = state.get_stats()
    loaded = [c for c, s in stats.items() if s["loaded"]]
    return {
        "status": "ok",
        "loaded_competencies": loaded,
        "indexes": stats,
    }

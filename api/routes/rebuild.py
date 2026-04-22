from __future__ import annotations

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from core.settings import get_settings
from services.rebuild import rebuild_index

router = APIRouter(tags=["rebuild"])


class RebuildRequest(BaseModel):
    use_gpu_model: bool = False
    model_service_url: str | None = None


@router.post("/api/v1/rebuild/{competency}")
async def rebuild(competency: str, req: RebuildRequest, x_api_key: str | None = Header(default=None)):
    s = get_settings()
    if s.admin_api_key and x_api_key != s.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin API key")

    if not s.catalog_path(competency).exists():
        raise HTTPException(status_code=404, detail=f"No catalog found for competency='{competency}'")

    result = await rebuild_index(
        competency=competency,
        use_gpu_model=req.use_gpu_model,
        model_service_url=req.model_service_url or s.model_service_url,
    )
    return result

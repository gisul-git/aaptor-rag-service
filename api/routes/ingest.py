from __future__ import annotations

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from core.settings import get_settings
from services.ingestion import ingest_entries

router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    entries: list[dict]


@router.post("/api/v1/ingest/{competency}")
async def ingest(competency: str, req: IngestRequest, x_api_key: str | None = Header(default=None)):
    s = get_settings()
    if s.admin_api_key and x_api_key != s.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin API key")

    if not req.entries:
        raise HTTPException(status_code=400, detail="No entries provided")

    result = await ingest_entries(competency=competency, entries=req.entries)
    return result

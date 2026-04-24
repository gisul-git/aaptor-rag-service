from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.retrieval import search

router = APIRouter(tags=["retrieve"])


class RetrieveRequest(BaseModel):
    competency: str
    topic: str
    difficulty: str = "Medium"
    concepts: list[str] = []
    top_k: int = 5


@router.post("/api/v1/retrieve")
async def retrieve(req: RetrieveRequest):
    if req.competency not in ["aiml", "dsa", "devops", "data_engineering", "design", "prompt_engineering", "cloud", "fullstack"]:
        raise HTTPException(status_code=400, detail=f"Unknown competency: {req.competency}")

    result = search(
        competency=req.competency,
        topic=req.topic,
        difficulty=req.difficulty,
        concepts=req.concepts,
        top_k=req.top_k,
    )

    if result is None:
        raise HTTPException(status_code=404, detail=f"No match found for competency='{req.competency}' topic='{req.topic}'")

    return result

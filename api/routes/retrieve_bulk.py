from __future__ import annotations

import random
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core import state
from core.state import COMPETENCIES

router = APIRouter(tags=["retrieve-bulk"])


class BulkRetrieveRequest(BaseModel):
    competency: str
    topic: str
    difficulty: str = "Medium"
    concepts: list[str] = []
    count: int = 5  # number of different problems to return


@router.post("/api/v1/retrieve/bulk")
async def retrieve_bulk(req: BulkRetrieveRequest):
    """
    Return N different matching problems for bulk question generation.
    Uses FAISS to find top candidates, then returns `count` unique ones.
    Falls back to random sampling from catalog if FAISS doesn't have enough.
    """
    if req.competency not in COMPETENCIES:
        raise HTTPException(status_code=400, detail=f"Unknown competency: {req.competency}")

    count = max(1, min(req.count, 50))
    results = _bulk_search(req.competency, req.topic, req.difficulty, req.concepts, count)

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No matches found for competency='{req.competency}' topic='{req.topic}'"
        )

    return {
        "competency": req.competency,
        "count": len(results),
        "matches": results,
    }


def _bulk_search(
    competency: str,
    topic: str,
    difficulty: str,
    concepts: list[str],
    count: int,
) -> list[dict]:
    """Return up to `count` unique matching problems."""
    catalog = state.catalogs.get(competency, [])
    if not catalog:
        return []

    difficulty_lower = difficulty.lower()
    results = []
    seen_titles = set()

    # Try FAISS first
    if competency in state.indexes:
        try:
            query = f"{topic} {' '.join(concepts)} {difficulty}"
            vec = state.embed_model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)

            # Search large pool
            search_k = min(count * 20, state.indexes[competency].ntotal)
            scores, indices = state.indexes[competency].search(vec, search_k)
            meta = state.metadata[competency]

            for idx, score in zip(indices[0], scores[0]):
                if len(results) >= count:
                    break
                if idx < 0 or idx >= len(meta):
                    continue
                if float(score) < 0.2:
                    continue

                entry_meta = meta[idx]
                meta_diff = entry_meta.get("difficulty", "")
                if meta_diff:
                    if isinstance(meta_diff, list):
                        meta_diff = meta_diff[0] if meta_diff else ""
                    if meta_diff.lower() != difficulty_lower:
                        continue

                # Get full entry
                stored_index = entry_meta.get("index", idx)
                entry_id = entry_meta.get("id", "")
                full_entry = None
                if entry_id:
                    full_entry = next((e for e in catalog if e.get("id") == entry_id), None)
                if full_entry is None and stored_index < len(catalog):
                    full_entry = catalog[stored_index]

                if full_entry:
                    title = full_entry.get("title", full_entry.get("name", ""))
                    if title not in seen_titles:
                        seen_titles.add(title)
                        results.append({
                            "matched": full_entry,
                            "score": round(float(score), 4),
                            "method": "faiss",
                        })
        except Exception as e:
            pass  # fall through to random sampling

    # If not enough from FAISS, fill with random samples from catalog
    if len(results) < count:
        diff_candidates = [
            e for e in catalog
            if str(e.get("difficulty", "")).lower() == difficulty_lower
            and e.get("title", e.get("name", "")) not in seen_titles
            and e.get("public_testcases")  # only problems with test cases
        ]
        random.shuffle(diff_candidates)
        for entry in diff_candidates:
            if len(results) >= count:
                break
            title = entry.get("title", entry.get("name", ""))
            if title not in seen_titles:
                seen_titles.add(title)
                results.append({
                    "matched": entry,
                    "score": 0.0,
                    "method": "random_sample",
                })

    return results

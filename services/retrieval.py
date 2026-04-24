from __future__ import annotations

import logging
import numpy as np
from typing import Any

from core import state

logger = logging.getLogger(__name__)


def search(
    competency: str,
    topic: str,
    difficulty: str,
    concepts: list[str],
    top_k: int = 5,
) -> dict | None:
    """Search FAISS index for best matching entry."""

    if competency not in state.indexes:
        logger.warning("No index loaded for competency='%s'", competency)
        return _keyword_fallback(competency, topic, difficulty, concepts)

    query = f"{topic} {' '.join(concepts)} {difficulty}"

    try:
        vec = state.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = state.indexes[competency].search(vec, top_k * 3)
        meta = state.metadata[competency]
        catalog = state.catalogs[competency]

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(meta):
                continue
            entry_meta = meta[idx]
            # Handle difficulty as list or string
            meta_diff = entry_meta.get("difficulty", "")
            if isinstance(meta_diff, list):
                meta_diff = meta_diff[0] if meta_diff else ""
            if meta_diff.lower() != difficulty.lower():
                continue
            # Find full catalog entry by id or by index
            entry_id = entry_meta.get("id", "")
            if entry_id:
                full_entry = next((e for e in catalog if e.get("id") == entry_id), None)
            else:
                # DSA catalog has no id — match by index directly
                full_entry = catalog[idx] if idx < len(catalog) else None
            if full_entry:
                logger.info(
                    "FAISS match: '%s' score=%.3f competency='%s'",
                    full_entry.get("name") or full_entry.get("title", ""), float(score), competency
                )
                return {
                    "matched": full_entry,
                    "score": round(float(score), 4),
                    "method": "faiss",
                    "competency": competency,
                }
    except Exception as e:
        logger.error("FAISS search failed for '%s': %s", competency, e)

    return _keyword_fallback(competency, topic, difficulty, concepts)


def _keyword_fallback(
    competency: str,
    topic: str,
    difficulty: str,
    concepts: list[str],
) -> dict | None:
    """Keyword-based fallback when FAISS is unavailable or returns no match."""
    catalog = state.catalogs.get(competency, [])
    if not catalog:
        return None

    keywords = [topic.lower()] + [c.lower() for c in concepts]
    difficulty_lower = difficulty.lower()

    for entry in catalog:
        # Handle difficulty as list or string
        entry_diff = entry.get("difficulty", "")
        if isinstance(entry_diff, list):
            entry_diff = entry_diff[0] if entry_diff else ""
        if str(entry_diff).lower() != difficulty_lower:
            continue
        searchable = " ".join([
            entry.get("name", ""),
            entry.get("title", ""),
            entry.get("description", ""),
            entry.get("problem_description", ""),
            entry.get("context", ""),          # DevOps
            entry.get("concept", ""),          # Cloud
            entry.get("core_idea", ""),        # Cloud
            entry.get("service", ""),          # Cloud
            " ".join(entry.get("tags", [])),
            " ".join(entry.get("topics", [])),
        ]).lower()
        if any(kw in searchable for kw in keywords):
            logger.info("Keyword match: '%s' competency='%s'", entry.get("name") or entry.get("title"), competency)
            return {
                "matched": entry,
                "score": 0.0,
                "method": "keyword",
                "competency": competency,
            }

    return None

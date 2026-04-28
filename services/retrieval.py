from __future__ import annotations

import logging
import numpy as np
from typing import Any

from core import state

logger = logging.getLogger(__name__)

# Minimum cosine similarity score to accept a FAISS match
_MIN_SCORE = 0.3


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

        # Search larger pool to increase chance of finding difficulty + concept match
        search_k = min(top_k * 20, state.indexes[competency].ntotal)
        scores, indices = state.indexes[competency].search(vec, search_k)
        meta = state.metadata[competency]
        catalog = state.catalogs[competency]

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(meta):
                continue

            # Skip low-confidence matches
            if float(score) < _MIN_SCORE:
                continue

            entry_meta = meta[idx]

            # Difficulty filter — skip if no difficulty in metadata (e.g. cloud)
            meta_diff = entry_meta.get("difficulty", "")
            if meta_diff:
                if isinstance(meta_diff, list):
                    meta_diff = meta_diff[0] if meta_diff else ""
                if meta_diff.lower() != difficulty.lower():
                    continue

            # Find full catalog entry (need it for concept filtering)
            stored_index = entry_meta.get("index", idx)
            entry_id = entry_meta.get("id", "")

            full_entry = None
            if entry_id:
                full_entry = next((e for e in catalog if e.get("id") == entry_id), None)
            if full_entry is None and stored_index < len(catalog):
                full_entry = catalog[stored_index]

            if not full_entry:
                continue

            # Concept filter (e.g. sql_category) — if concepts provided, at least one must match
            if concepts:
                entry_category = full_entry.get("sql_category", "").lower()
                entry_tags = [t.lower() for t in full_entry.get("tags", [])]
                entry_topics = [t.lower() for t in full_entry.get("topics", [])]
                concepts_lower = [c.lower() for c in concepts]
                
                # Check if any concept matches sql_category, tags, or topics
                if not any(
                    c == entry_category or c in entry_tags or c in entry_topics
                    for c in concepts_lower
                ):
                    continue

            # Return matched entry
            if full_entry:
                logger.info(
                    "FAISS match: '%s' score=%.3f competency='%s'",
                    full_entry.get("name") or full_entry.get("title") or full_entry.get("concept", ""),
                    float(score),
                    competency,
                )
                return {
                    "matched": full_entry,
                    "score": round(float(score), 4),
                    "method": "faiss",
                    "competency": competency,
                }

        logger.info("No FAISS match above threshold for '%s' difficulty='%s' — trying keyword", competency, difficulty)

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
        # Handle difficulty as list or string — skip filter if entry has no difficulty
        entry_diff = entry.get("difficulty", "")
        if entry_diff:
            if isinstance(entry_diff, list):
                entry_diff = entry_diff[0] if entry_diff else ""
            if str(entry_diff).lower() != difficulty_lower:
                continue
        searchable = " ".join([
            entry.get("name", ""),
            entry.get("title", ""),
            entry.get("description", ""),
            entry.get("problem_description", ""),
            entry.get("context", ""),
            entry.get("concept", ""),
            entry.get("core_idea", ""),
            entry.get("service", ""),
            entry.get("sql_category", ""),
            " ".join(entry.get("tags", [])),
            " ".join(entry.get("topics", [])),
        ]).lower()

        # If concepts provided, also require at least one concept match on sql_category/tags/topics
        if concepts:
            concepts_lower = [c.lower() for c in concepts]
            entry_category = entry.get("sql_category", "").lower()
            entry_tags = [t.lower() for t in entry.get("tags", [])]
            entry_topics = [t.lower() for t in entry.get("topics", [])]
            if not any(c == entry_category or c in entry_tags or c in entry_topics for c in concepts_lower):
                continue

        if any(kw in searchable for kw in keywords):
            logger.info("Keyword match: '%s' competency='%s'", entry.get("name") or entry.get("title"), competency)
            return {
                "matched": entry,
                "score": 0.0,
                "method": "keyword",
                "competency": competency,
            }

    return None

"""
Ingestion service — adds new entries to MongoDB catalog and rebuilds FAISS index.
MongoDB is always the source of truth.
"""
from __future__ import annotations

import logging

from db import mongo
from services.rebuild import rebuild_index

logger = logging.getLogger(__name__)


async def ingest_entries(competency: str, entries: list[dict]) -> dict:
    """
    Add new entries to MongoDB catalog and rebuild FAISS index.
    Deduplication is handled by MongoDB upsert (_id based).
    """
    # Upsert into MongoDB
    upserted = mongo.upsert_entries(competency, entries)
    total = mongo.count(competency)

    logger.info(
        "Ingested %d entries into '%s' catalog (total: %d)",
        upserted, competency, total
    )

    # Rebuild FAISS index from updated MongoDB catalog
    rebuild_result = await rebuild_index(competency=competency)

    return {
        "competency": competency,
        "upserted": upserted,
        "total_catalog": total,
        **rebuild_result,
    }

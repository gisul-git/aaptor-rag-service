"""
MongoDB catalog storage for aaptor-rag-service.

MongoDB is the source of truth for catalog data.
FAISS index is always rebuilt from MongoDB.
"""
from __future__ import annotations

import logging
from typing import Any

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

_client: MongoClient | None = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        from core.settings import get_settings
        _client = MongoClient(get_settings().mongodb_uri)
    return _client


def _collection(competency: str) -> Collection:
    from core.settings import get_settings
    db = _get_client()[get_settings().mongodb_db_name]
    return db[f"{competency}_catalog"]


def ensure_indexes(competency: str) -> None:
    """Create MongoDB indexes for fast queries."""
    col = _collection(competency)
    col.create_index("competency")
    col.create_index("difficulty")
    col.create_index("tags")
    col.create_index("domain")
    logger.info("MongoDB indexes ensured for '%s'", competency)


def upsert_entries(competency: str, entries: list[dict]) -> int:
    """
    Upsert catalog entries into MongoDB.
    Uses _id as the unique key — inserts new, updates existing.
    Returns count of upserted documents.
    """
    if not entries:
        return 0

    def _sanitize(obj):
        """Recursively sanitize values for MongoDB compatibility."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(i) for i in obj]
        if isinstance(obj, int):
            # MongoDB max int is 8 bytes (int64 max = 9223372036854775807)
            if obj > 9223372036854775807 or obj < -9223372036854775808:
                return str(obj)
        return obj

    col = _collection(competency)
    ops = []
    for entry in entries:
        doc = _sanitize(dict(entry))
        doc["competency"] = competency
        doc_id = doc.pop("id", None) or str(doc.get("title", "")).lower().replace(" ", "-")
        ops.append(UpdateOne(
            {"_id": doc_id},
            {"$set": doc},
            upsert=True,
        ))

    if ops:
        result = col.bulk_write(ops)
        count = result.upserted_count + result.modified_count
        logger.info("Upserted %d entries into '%s_catalog'", count, competency)
        return count
    return 0


def load_all(competency: str) -> list[dict]:
    """Load all catalog entries for a competency from MongoDB."""
    col = _collection(competency)
    docs = list(col.find({}, {"_id": 0}))
    logger.info("Loaded %d entries from MongoDB '%s_catalog'", len(docs), competency)
    return docs


def get_by_id(competency: str, doc_id: str) -> dict | None:
    """Get a single catalog entry by ID."""
    col = _collection(competency)
    doc = col.find_one({"_id": doc_id}, {"_id": 0})
    return doc


def count(competency: str) -> int:
    return _collection(competency).count_documents({})


def sync_from_json(competency: str, catalog: list[dict]) -> int:
    """
    Sync entire catalog from JSON list into MongoDB.
    Used during initial migration and rebuild.
    """
    return upsert_entries(competency, catalog)

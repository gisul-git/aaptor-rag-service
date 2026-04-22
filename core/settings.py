from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service
    host: str = "0.0.0.0"
    port: int = 7003
    admin_api_key: str = ""
    allowed_origins: list[str] = ["*"]

    # Data root
    data_dir: Path = Path("data")

    # Embedding model (CPU)
    embed_model_name: str = "all-MiniLM-L6-v2"

    # GPU model service (used for DSA enrichment during rebuild)
    model_service_url: str = "http://127.0.0.1:7001"

    # MongoDB (local on VM)
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "rag_db"

    class Config:
        env_file = ".env"

    def competency_dir(self, competency: str) -> Path:
        return self.data_dir / competency

    def catalog_path(self, competency: str) -> Path:
        return self.competency_dir(competency) / "catalog.json"

    def faiss_path(self, competency: str) -> Path:
        return self.competency_dir(competency) / "faiss.index"

    def metadata_path(self, competency: str) -> Path:
        return self.competency_dir(competency) / "metadata.json"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

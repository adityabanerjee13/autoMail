"""Runtime configuration. Everything from the environment, nothing hardcoded."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- storage ---------------------------------------------------------
    database_url: str = "postgresql+psycopg://triage:triage@127.0.0.1:5432/triage"

    # ---- inference -------------------------------------------------------
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_model_id: str = "Qwen3-8B-AWQ"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 120.0
    # Qwen3 and other reasoning models emit a <think> block by default.
    # Under a JSON grammar they cannot, so they emit whitespace filler
    # until max_tokens instead -- a hang that looks like a bad schema.
    llm_disable_thinking: bool = True

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"

    # ---- gmail -----------------------------------------------------------
    gmail_credentials_path: Path = Path("data/secrets/credentials.json")
    gmail_token_path: Path = Path("data/secrets/token.json")
    gmail_user_id: str = "me"
    gcp_project_id: str = ""
    pubsub_subscription: str = ""
    pubsub_topic: str = ""
    history_poll_interval_s: int = 60

    # ---- imap backfill ---------------------------------------------------
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    imap_username: str = ""
    imap_folder: str = "[Gmail]/All Mail"
    imap_concurrency: int = 16

    # ---- pipeline --------------------------------------------------------
    prompt_version: str = "v1"
    schema_version: str = "v1"
    enable_fewshot: bool = False
    fewshot_k: int = 5
    enable_dedup: bool = True
    max_body_tokens: int = 800
    worker_concurrency: int = 4
    job_max_attempts: int = 5
    job_stuck_timeout_s: int = 600

    # ---- api -------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    log_level: str = "INFO"

    prompt_dir: Path = Field(default=REPO_ROOT / "src" / "triage" / "llm" / "prompts")

    @field_validator("database_url")
    @classmethod
    def _validate_dsn(cls, v: str) -> str:
        # Catches a plain postgresql:// URL, which SQLAlchemy would route to
        # psycopg2 -- a driver this project does not install.
        PostgresDsn(v.replace("+psycopg", ""))
        if "+psycopg" not in v:
            raise ValueError("DATABASE_URL must use the postgresql+psycopg:// driver")
        return v

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else (REPO_ROOT / p)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

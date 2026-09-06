"""Sentence embeddings for dedup neighbours and few-shot retrieval.

Defaults to CPU. Do not move this to the GPU without measuring: the 16 GB card
is sized for the generator plus its KV cache, and at this message volume the
embedding pass is not the bottleneck.

The model is loaded lazily on first use so that processes which never embed
(the API, the sync daemon) do not pay several hundred MB of RSS -- 18 GB of
system RAM is the tighter constraint on this box, not VRAM.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from triage.config import Settings, get_settings

log = logging.getLogger(__name__)

_model: Any = None
_lock = threading.Lock()


def _load(settings: Settings):
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            log.info(
                "loading embedding model %s on %s",
                settings.embedding_model,
                settings.embedding_device,
            )
            _model = SentenceTransformer(
                settings.embedding_model, device=settings.embedding_device
            )
    return _model


def embed_sync(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    settings = settings or get_settings()
    if not texts:
        return []
    model = _load(settings)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,  # cosine distance in pgvector assumes this
        show_progress_bar=False,
        batch_size=8,
    )
    out = [list(map(float, v)) for v in vectors]
    dim = len(out[0])
    if dim != settings.embedding_dim:
        raise RuntimeError(
            f"embedding model produced {dim} dims but messages.embedding is "
            f"vector({settings.embedding_dim}); this needs a migration, not a config change"
        )
    return out


async def embed(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """Off-thread wrapper: encoding is blocking and would stall the worker loop."""
    return await asyncio.to_thread(embed_sync, texts, settings)


def embedding_text(subject: str | None, body: str, *, limit: int = 2000) -> str:
    """What actually gets embedded.

    Subject first because it survives truncation, and truncation is common:
    embedding models have far shorter context than the generator.
    """
    return f"{(subject or '').strip()}\n\n{body.strip()}"[:limit]

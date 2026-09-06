# On-prem Gmail triage pipeline — implementation handoff

## Objective

Ingest a single personal Gmail mailbox, classify every message for importance and
category, extract a small set of structured fields, and persist every judgment for
later review and model improvement. All inference runs locally. No data leaves the
machine except the authenticated Gmail API calls required to fetch mail.

## Hardware constraints

- GPU: 16 GB VRAM
- System RAM: 18 GB — this is the tighter constraint, not VRAM
- Single machine, no external services beyond Google's API

## Non-negotiable decisions already made

Do not revisit these without asking. They were decided deliberately.

| Decision | Rationale |
| --- | --- |
| No Redis | Postgres `SKIP LOCKED` + `LISTEN/NOTIFY` covers queueing at this volume. One stateful service instead of two. |
| No rules pre-filter stage | Removed by request. Every message goes to the LLM. |
| No supervised classifier stage | Removed by request. Phase 2 at the earliest. |
| One LLM call per message | Classification and extraction share a single guided-JSON schema. Two calls would double GPU time for no added information. |
| Append-only classification history | Never update in place. Rows are versioned by model and prompt. |
| Pydantic is validation, not orchestration | Orchestration is plain Python functions called in sequence. Do not introduce LangChain, LangGraph, or an agent loop. The pipeline is deterministic. |

---

## Stack

| Layer | Choice |
| --- | --- |
| Ingestion | `google-api-python-client` (Gmail API v1), `imapclient` for backfill |
| Storage | PostgreSQL 16 + `pgvector` |
| Queue | Postgres table, `SELECT ... FOR UPDATE SKIP LOCKED`, `LISTEN/NOTIFY` |
| Migrations | Alembic |
| Inference | vLLM, OpenAI-compatible endpoint |
| Model | Qwen3-8B AWQ 4-bit (start here; 14B AWQ is the upgrade path) |
| Embeddings | `bge-m3` or `e5-base` via sentence-transformers |
| Schemas | Pydantic v2 |
| API / UI | FastAPI + Jinja + HTMX |
| Scheduling | APScheduler |
| HTML/MIME parsing | `selectolax`, `talon`, stdlib `email` |
| Process management | systemd units (preferred over Docker for the GPU process) |

### VRAM budget

| Component | Approx. |
| --- | --- |
| Qwen3-8B AWQ weights | 5–6 GB |
| KV cache (8k ctx, small batch) | 2–3 GB |
| Embedding model | 0.5–1 GB |
| Headroom | ~4 GB |

Set `--gpu-memory-utilization 0.80` on vLLM. Do not run embeddings and generation
on the GPU simultaneously without measuring first — embeddings on CPU are
acceptable at this volume.

---

## Gmail access

### Auth

- OAuth 2.0, **Desktop app** client type, created in Google Cloud Console.
- Scope: `https://www.googleapis.com/auth/gmail.readonly` (a restricted scope).
- **Set the OAuth consent screen publishing status to "In production."** If left in
  "Testing," refresh tokens expire after 7 days and the daemon dies silently.
- Expect an "unverified app" interstitial on first consent. Click through. Full
  verification is only needed if this is ever distributed to other users.
- Refresh tokens carrying Gmail scopes are **invalidated when the Google account
  password changes**. Build a visible "reconnect account" path in the UI from day one.
- Store the refresh token at `data/secrets/`, mode 0600, owned by the service user.

### Sync model

Incremental sync uses `users.history.list` with a persisted `historyId` watermark.
Do not implement IMAP UID tracking for the steady-state path.

Google retains roughly one week of history. If `history.list` returns 404 the
watermark is stale — fall back to a bounded `messages.list` resync and re-checkpoint.

### Quota (changed 1 May 2026 — verify before sizing the backfill)

- 1,200,000 quota units/minute per project
- 6,000 quota units/minute per user per project
- 80,000,000 units/day per project billing threshold

Per-method: `history.list` 2, `messages.list` 5, `messages.get` 20, `threads.get` 40.

At 6,000 units/min, `messages.get` caps you at ~300 message bodies per minute.
`fields=` partial responses and HTTP batching reduce payload and round-trips but
**not** quota cost.

### Backfill

Use **IMAP** for the initial historical import, not the Gmail API — IMAP is
bandwidth-limited (~2.5 GB/day) rather than quota-unit-limited, which is far cheaper
for bulk reads.

Gmail IMAP trap: labels are exposed as folders, so the same message appears in
`INBOX`, `[Gmail]/All Mail`, and every label folder. Sync `[Gmail]/All Mail` only, or
deduplicate on `X-GM-MSGID`. Use the `X-GM-EXT-1` extensions (`X-GM-MSGID`,
`X-GM-THRID`, `X-GM-LABELS`).

---

## Data model

### `messages`

Immutable record of what arrived.

```
id              bigserial PK
gmail_id        text UNIQUE NOT NULL
thread_id       text NOT NULL
message_id_hdr  text                    -- RFC 822 Message-ID
from_addr       text NOT NULL
to_addrs        text[]
cc_addrs        text[]
subject         text
body_clean      text                    -- dequoted, HTML stripped
snippet         text
labels          text[]                  -- Gmail's own labels
headers         jsonb                   -- selected headers only
fingerprint     text                    -- template dedup key
embedding       vector(1024)
internal_date   timestamptz NOT NULL
created_at      timestamptz DEFAULT now()
```

Index: `gmail_id`, `thread_id`, `from_addr`, `fingerprint`, HNSW on `embedding`.

### `classifications`

Append-only. Never `UPDATE`. Latest row per `message_id` wins.

```
id              bigserial PK
message_id      bigint NOT NULL REFERENCES messages(id)
is_important    boolean NOT NULL
category        text NOT NULL
payload         jsonb NOT NULL          -- full parsed Triage object
raw_response    text NOT NULL           -- exactly what the model returned
model_id        text NOT NULL           -- 'Qwen3-8B-AWQ'
prompt_version  text NOT NULL           -- 'v1'
schema_version  text NOT NULL
source          text NOT NULL           -- 'llm' | 'dedup' | 'human'
input_tokens    int
latency_ms      int
created_at      timestamptz DEFAULT now()
```

The three version columns are load-bearing. Without them the accumulated dataset is a
mixture of incompatible labellers and is worthless for phase 2 training.

Keep `raw_response` even though `payload` is also stored — schema validation failures
and truncated generations cannot be reconstructed from parsed output.

Human corrections are inserted here with `source='human'`.

### `jobs`

```
id              bigserial PK
message_id      bigint NOT NULL REFERENCES messages(id)
status          text NOT NULL           -- 'pending'|'running'|'done'|'dead'
attempts        int NOT NULL DEFAULT 0
run_after       timestamptz NOT NULL DEFAULT now()
last_error      text
started_at      timestamptz
created_at      timestamptz DEFAULT now()
```

Partial index on `(run_after)` where `status = 'pending'`.

### `sync_state`

Single-row table holding `history_id`, `watch_expiry`, `last_reconcile_at`.

---

## Queue implementation

Claim:

```sql
UPDATE jobs SET status = 'running', started_at = now()
WHERE id = (
  SELECT id FROM jobs
  WHERE status = 'pending' AND run_after <= now()
  ORDER BY id
  LIMIT 1
  FOR UPDATE SKIP LOCKED
)
RETURNING *;
```

Enqueue and the corresponding `messages` insert must happen **in the same
transaction** — that transactional guarantee is the reason Redis was dropped.

Wake-up: the enqueueing transaction issues `NOTIFY triage_jobs`. Workers hold a
dedicated connection on `LISTEN triage_jobs`, drain the queue on wake, then block
again. Do not poll.

---

## Schemas

`src/triage/taxonomy.py` holds the category enum. It is the **single source of truth**
and feeds three consumers: the Pydantic model, the JSON Schema passed to vLLM, and the
review UI dropdown. If these drift you get silent misclassification that is very hard
to debug. Derive all three from the enum; never hardcode the list twice.

`src/triage/llm/schemas.py`:

```python
class Triage(BaseModel):
    is_important: bool
    importance_confidence: Literal["low", "medium", "high"]
    category: Category
    category_confidence: Literal["low", "medium", "high"]
    action_required: bool
    deadline: date | None
    reason: str = Field(max_length=200)
```

Pass `Triage.model_json_schema()` to vLLM's `guided_json`. Do not parse free text.

Notes on field choices:

- `reason` is for human debugging of misclassifications and for discovering ambiguous
  category definitions. It is not decorative.
- Confidence is a coarse bucket, not a float. LLMs produce badly calibrated numbers but
  usable buckets. `low` on either field routes the message to the review queue.

---

## Runner stages

`pipeline/runner.py`, executed per message. Keep this function boring. If it starts
accumulating branching logic, the stage boundaries are wrong — do not reach for an
orchestration framework.

1. **Load** — fetch message row and thread context.
2. **Idempotency guard** — if a `classifications` row already exists for this
   `message_id` at the current `model_id` + `prompt_version`, return early.
3. **Dedup check** — compute template fingerprint (normalized sender + subject with
   digits and dates stripped). On a high-confidence prior hit, copy that result with
   `source='dedup'` and jump to step 8.
4. **Features** — sender history (have we ever replied to this sender), header signals,
   To vs Cc vs Bcc position. Passed into the prompt as context.
5. **Prepare input** — subject + sender + dequoted body truncated to ~800 tokens.
   Never send full quoted thread history; it inflates latency without improving
   accuracy.
6. **Few-shot retrieval** *(config-gated, off initially)* — pgvector nearest neighbours
   over already-labelled messages, preferring rows with `source='human'`. Inject as
   examples. Leave disabled until several hundred corrected labels exist; the pool is
   empty on day one.
7. **LLM call + validate** — vLLM with `guided_json`. Parse into `Triage`. On schema
   failure retry once at temperature 0, then dead-letter the job.
8. **Persist** — append to `classifications` with `raw_response`, `model_id`,
   `prompt_version`, `schema_version`, `source`, `input_tokens`, `latency_ms`.
9. **Post-actions** — store embedding and fingerprint on the message row; flag for
   review if either confidence is `low`.
10. **Complete** — mark job done. On exception, increment `attempts` and set
    `run_after` with exponential backoff; dead-letter past a threshold.

---

## Scheduling

Three independent mechanisms. None of them is a cron loop over the whole pipeline.

**Ingestion — Pub/Sub pull subscription.** `users.watch()` registers the mailbox
against a Pub/Sub topic; the sync daemon holds an outbound long-poll. Pull (not push)
means no public HTTPS endpoint and no inbound firewall rules, which is what makes this
viable on-prem. Fallback if Pub/Sub is skipped: poll `history.list` every 60s.

**Worker — event-driven.** `LISTEN triage_jobs`. Run 2–4 worker processes with an
`asyncio.Semaphore` capping concurrent vLLM requests.

**Periodic — APScheduler in `scheduler.py`:**

| Job | Interval |
| --- | --- |
| Renew Gmail `watch()` | daily (expiry is 7 days — do not schedule weekly) |
| Reconciliation sweep via `history.list` | hourly |
| Requeue jobs stuck in `running` past timeout | 5 min |
| Retry dead-lettered jobs | hourly |
| Eval against `golden.jsonl` | weekly |
| Prune old fingerprints | weekly |

The hourly reconciliation is the most important entry. Pub/Sub is at-least-once but not
guaranteed-delivery; without an independent watermark sweep you will silently lose
emails and notice weeks later.

---

## Folder structure

```
email-triage/
├── pyproject.toml
├── alembic.ini
├── .env.example
├── docker-compose.yml
├── README.md
│
├── deploy/
│   ├── systemd/
│   │   ├── triage-sync.service
│   │   ├── triage-worker.service
│   │   └── triage-api.service
│   └── vllm.env
│
├── migrations/
│   └── versions/
│
├── data/                          # gitignored
│   ├── secrets/
│   ├── models/
│   └── labels/
│
├── src/triage/
│   ├── config.py
│   ├── taxonomy.py
│   ├── schemas.py
│   │
│   ├── db/
│   │   ├── engine.py
│   │   ├── models.py
│   │   ├── queue.py
│   │   └── repo.py
│   │
│   ├── ingest/
│   │   ├── base.py
│   │   ├── gmail_api.py
│   │   ├── imap.py
│   │   ├── auth.py
│   │   └── daemon.py
│   │
│   ├── parse/
│   │   ├── mime.py
│   │   ├── html.py
│   │   ├── dequote.py
│   │   └── threading.py
│   │
│   ├── pipeline/
│   │   ├── runner.py
│   │   ├── features.py
│   │   ├── dedup.py
│   │   └── stage1_llm.py
│   │
│   ├── llm/
│   │   ├── client.py
│   │   ├── schemas.py
│   │   ├── fewshot.py
│   │   └── prompts/
│   │       └── classify_v1.jinja
│   │
│   ├── scheduler.py
│   ├── worker.py
│   ├── backfill.py
│   └── api/
│       ├── app.py
│       ├── routes/
│       │   ├── messages.py
│       │   ├── review.py
│       │   └── health.py
│       └── templates/
│
├── eval/
│   ├── golden.jsonl
│   ├── run_eval.py
│   └── report.py
│
└── tests/
    ├── fixtures/emails/
    ├── test_parse.py
    └── test_queue.py
```

### Module conventions

- `ingest/base.py` defines a `MailSource` protocol with `fetch_since(checkpoint)` and
  `backfill(before)`. Gmail API and IMAP both implement it. Nothing above `ingest/`
  knows which is running.
- `parse/` is pure and deterministic — same bytes in, same text out. No network, no
  database. This is where subtle bugs live, so it must be unit-testable against `.eml`
  fixtures alone.
- `db/repo.py` returns **Pydantic models, not SQLAlchemy rows**. If ORM objects leak
  upward you get lazy-loading and detached-instance errors inside worker code.
- Prompts are versioned **by filename**, not git history. A stored row saying
  `prompt_version='v3'` must map to a file you can open.

---

## Processes

Five, managed by systemd:

1. `vllm serve` — owns the GPU
2. `postgres` — with `pgvector` extension enabled
3. `triage-sync` — Pub/Sub pull, writes message rows, enqueues jobs
4. `triage-worker` — 2–4 instances, LISTEN/NOTIFY, runs the pipeline
5. `triage-api` — FastAPI + review UI

Docker Compose is acceptable for Postgres. Prefer bare systemd for vLLM — GPU
passthrough in containers adds a debugging layer that buys nothing on a single machine.

Set Postgres `shared_buffers` to ~2 GB and keep it modest. Postgres, the worker,
sentence-transformers, and MIME parsing all compete for the same 18 GB.

---

## Configuration

`config.py` uses `pydantic-settings`. Everything from env, nothing hardcoded.

```
DATABASE_URL
VLLM_BASE_URL
VLLM_MODEL_ID
EMBEDDING_MODEL
GMAIL_CREDENTIALS_PATH
GMAIL_TOKEN_PATH
PUBSUB_SUBSCRIPTION
PROMPT_VERSION
SCHEMA_VERSION
ENABLE_FEWSHOT              # default false
MAX_BODY_TOKENS             # default 800
WORKER_CONCURRENCY          # default 4
```

---

## Evaluation

`eval/golden.jsonl` is the most valuable artifact in the repository. **Commit it.**

- 300–500 hand-labelled emails, held out from everything.
- Primary metric is **recall on `is_important`**, not accuracy. A missed important
  email costs far more than a false positive. Tune toward over-escalation.
- Secondary: per-category precision and recall, plus a confusion matrix — this is how
  you find category definitions that overlap.
- `run_eval.py` must be runnable against any `(model_id, prompt_version)` pair so
  changes can be compared rather than argued about.

---

## Build order

1. Ingestion + parse + store, no inference at all. Verify re-ingestion produces no
   duplicates and that a restart resumes cleanly from the watermark.
2. Postgres queue and worker skeleton that does nothing but mark jobs done.
3. vLLM up, `llm/client.py` with guided decoding, single message end to end.
4. **Before the backfill:** run 50 messages, read every `reason` field manually. Fix
   ambiguous category definitions and prompt instructions the model is ignoring.
     Doing this at 50 messages costs an hour; discovering it at 50,000 costs the run.
5. Review UI and `golden.jsonl`.
6. Backfill over IMAP with 16–32 concurrent requests, overnight.
7. Dedup fingerprinting, then few-shot retrieval once labels accumulate.

---

## Deferred to phase 2

Once several thousand stored judgments exist, train a small supervised classifier on
them (embeddings + logistic regression or LightGBM) and demote the LLM to handling only
low-confidence cases. This inverts the pipeline: LLM-everything now, LLM-as-fallback
later. There is currently no `ml/` module — training will need a home when this starts.

Do not build this yet. It is blocked on data that does not exist.
"""Score a (model_id, prompt_version) pair against the golden set.

    python -m eval.run_eval
    python -m eval.run_eval --model Qwen3-14B-AWQ --prompt-version v2
    python -m eval.run_eval --export-corrections   # grow golden.jsonl from the review UI

Deliberately does not touch the messages table: the golden rows carry their own
text, so any pair can be scored on any machine without the production database,
and re-running a month later scores the same inputs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from eval.report import Pair, Report, render, score  # noqa: F401 (render used by _write)
from triage.config import Settings, get_settings
from triage.db.engine import configure_event_loop_policy
from triage.llm.client import LLMClient, LLMSchemaError
from triage.pipeline.stage1_llm import truncate_to_tokens

log = logging.getLogger("eval")

GOLDEN = Path(__file__).parent / "golden.jsonl"

#: Below this the numbers are noise. The handoff calls for 300-500 rows; the
#: committed file starts far smaller than that on purpose (see README).
MIN_USEFUL_ROWS = 100

#: Feature values used for every golden row. Held constant so a change in
#: score is attributable to the model or the prompt, never to mailbox state
#: that has drifted since the row was labelled.
NEUTRAL_FEATURES = {
    "messages_from_sender": 0,
    "user_has_replied": False,
    "prior_important_from_sender": 0,
    "recipient_position": "unknown",
    "bulk_headers": "none",
    "automated_sender": False,
    "gmail_labels": "none",
    "has_attachments": False,
}


def load_golden(path: Path = GOLDEN, *, exclude_synthetic: bool = False) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"no golden set at {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        row = json.loads(line)
        if exclude_synthetic and row.get("synthetic"):
            continue
        rows.append(row)
    return rows


async def evaluate(
    rows: list[dict],
    *,
    settings: Settings | None = None,
    concurrency: int = 4,
) -> list[Pair]:
    settings = settings or get_settings()
    client = LLMClient(settings)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> Pair:
        expected = row["expected"]
        context = {
            "from_addr": row.get("from_addr", ""),
            "to_addrs": ", ".join(row.get("to_addrs", [])) or "(none)",
            "cc_addrs": ", ".join(row.get("cc_addrs", [])),
            "subject": row.get("subject") or "(no subject)",
            "body": truncate_to_tokens(row.get("body", ""), settings.max_body_tokens),
            "message_date": row.get("internal_date", "")[:10] or "unknown",
            "features": {**NEUTRAL_FEATURES, **row.get("features", {})},
            "thread": [],
            # Few-shot is off for eval regardless of config: the golden rows
            # would otherwise be scored against a retrieval pool that changes
            # every week, and two runs would not be comparable.
            "fewshot": [],
        }
        async with semaphore:
            try:
                result = await client.classify(context)
            except LLMSchemaError as exc:
                return Pair(
                    gmail_id=row.get("gmail_id", "?"),
                    subject=row.get("subject", ""),
                    expected_important=expected["is_important"],
                    predicted_important=False,
                    expected_category=expected["category"],
                    predicted_category="",
                    error=f"schema failure: {exc}",
                )
            except Exception as exc:  # noqa: BLE001
                return Pair(
                    gmail_id=row.get("gmail_id", "?"),
                    subject=row.get("subject", ""),
                    expected_important=expected["is_important"],
                    predicted_important=False,
                    expected_category=expected["category"],
                    predicted_category="",
                    error=repr(exc),
                )

        t = result.triage
        return Pair(
            gmail_id=row.get("gmail_id", "?"),
            subject=row.get("subject", ""),
            expected_important=expected["is_important"],
            predicted_important=t.is_important,
            expected_category=expected["category"],
            predicted_category=t.category.value,
            importance_confidence=t.importance_confidence,
            category_confidence=t.category_confidence,
            latency_ms=result.latency_ms,
            reason=t.reason,
        )

    return list(await asyncio.gather(*(one(r) for r in rows)))


async def run_and_report(
    *,
    settings: Settings | None = None,
    path: Path = GOLDEN,
    exclude_synthetic: bool = False,
    out_dir: Path | None = None,
) -> dict:
    settings = settings or get_settings()
    rows = load_golden(path, exclude_synthetic=exclude_synthetic)
    if len(rows) < MIN_USEFUL_ROWS:
        log.warning(
            "golden set has only %d rows; %d+ before these numbers mean anything",
            len(rows),
            MIN_USEFUL_ROWS,
        )
    if any(r.get("synthetic") for r in rows):
        log.warning(
            "golden set contains synthetic rows -- replace them with real hand-labelled "
            "mail before trusting a score"
        )

    pairs = await evaluate(rows, settings=settings)
    report = score(
        pairs, model_id=settings.vllm_model_id, prompt_version=settings.prompt_version
    )
    if out_dir:
        _write(report, out_dir, settings)
    return report.to_dict()


def _write(report: Report, out_dir: Path, settings: Settings) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = f"{settings.vllm_model_id}_{settings.prompt_version}_{stamp}".replace("/", "-")
    (out_dir / f"eval-{slug}.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / f"eval-{slug}.md").write_text(render(report), encoding="utf-8")
    log.info("wrote report to %s", out_dir / f"eval-{slug}.md")


async def export_corrections(path: Path, *, limit: int = 1000) -> int:
    """Append human-corrected messages to a jsonl file.

    This is how golden.jsonl grows to 300-500 rows: correct mail in the review
    UI, then export. The rows carry their own text so the file stays scoreable
    without the database.
    """
    from sqlalchemy import desc, select

    from triage.db.engine import sessionmaker_for
    from triage.db.models import ClassificationRow, MessageRow

    existing = set()
    if path.exists():
        existing = {
            json.loads(line)["gmail_id"]
            for line in path.read_text("utf-8").splitlines()
            if line.strip()
        }

    written = 0
    async with sessionmaker_for()() as session:
        rows = (
            await session.execute(
                select(MessageRow, ClassificationRow)
                .join(ClassificationRow, ClassificationRow.message_id == MessageRow.id)
                .where(ClassificationRow.source == "human")
                .order_by(desc(ClassificationRow.id))
                .limit(limit)
            )
        ).all()

    with path.open("a", encoding="utf-8") as fh:
        for message, classification in rows:
            if message.gmail_id in existing:
                continue
            fh.write(
                json.dumps(
                    {
                        "gmail_id": message.gmail_id,
                        "from_addr": message.from_addr,
                        "to_addrs": list(message.to_addrs or []),
                        "cc_addrs": list(message.cc_addrs or []),
                        "subject": message.subject,
                        "body": message.body_clean,
                        "internal_date": message.internal_date.isoformat(),
                        "expected": {
                            "is_important": classification.is_important,
                            "category": classification.category,
                            "action_required": bool(
                                (classification.payload or {}).get("action_required")
                            ),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            existing.add(message.gmail_id)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="override VLLM_MODEL_ID for this run")
    parser.add_argument("--prompt-version", help="override PROMPT_VERSION for this run")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--exclude-synthetic", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("data/labels"))
    parser.add_argument(
        "--export-corrections",
        action="store_true",
        help="append human-corrected messages to the golden file and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)-7s %(message)s")

    # export_corrections talks to Postgres; the rest does not, but setting
    # the policy unconditionally keeps the two paths identical.
    configure_event_loop_policy()

    if args.export_corrections:
        n = asyncio.run(export_corrections(args.golden))
        print(f"appended {n} corrected message(s) to {args.golden}")
        return

    settings = get_settings().model_copy(
        update={
            k: v
            for k, v in {
                "vllm_model_id": args.model,
                "prompt_version": args.prompt_version,
            }.items()
            if v
        }
    )

    report = asyncio.run(
        run_and_report(
            settings=settings,
            path=args.golden,
            exclude_synthetic=args.exclude_synthetic,
            out_dir=args.out_dir,
        )
    )
    # Human-readable to stdout, machine-readable (json + markdown) on disk.
    print(_render_from_dict(report))


def _render_from_dict(d: dict) -> str:
    lines = [
        f"{d['model_id']} / {d['prompt_version']}: {d['n']} scored, {d['errors']} errored",
        f"  important recall    {d['important_recall']:.3f}   <- primary",
        f"  important precision {d['important_precision']:.3f}",
        f"  category accuracy   {d['category_accuracy']:.3f}",
        f"  missed important    {len(d['missed_important'])}",
        f"  latency p50/p95     {d['latency_ms_p50']}ms / {d['latency_ms_p95']}ms",
    ]
    for m in d["missed_important"]:
        lines.append(f"    MISSED {m['gmail_id']} {m['subject']!r}: {m['reason']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

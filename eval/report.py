"""Scoring and report rendering. Pure functions over prediction pairs.

Primary metric is recall on ``is_important``, not accuracy. A missed important
email costs far more than a false positive, so the tuning direction is
over-escalation and accuracy would hide exactly the error that matters -- a
mailbox that is 95% unimportant scores 95% by predicting "no" every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from triage.taxonomy import category_values


@dataclass(slots=True)
class Pair:
    """One golden row and what the model said about it."""

    gmail_id: str
    subject: str
    expected_important: bool
    predicted_important: bool
    expected_category: str
    predicted_category: str
    importance_confidence: str = "unknown"
    category_confidence: str = "unknown"
    latency_ms: int = 0
    reason: str = ""
    error: str | None = None


@dataclass(slots=True)
class Binary:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass(slots=True)
class Report:
    model_id: str
    prompt_version: str
    n: int
    errors: int
    importance: Binary
    category_accuracy: float
    per_category: dict[str, Binary]
    confusion: dict[str, dict[str, int]]
    missed_important: list[Pair] = field(default_factory=list)
    latency_ms_p50: int = 0
    latency_ms_p95: int = 0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "n": self.n,
            "errors": self.errors,
            "important_recall": self.importance.recall,
            "important_precision": self.importance.precision,
            "important_f1": self.importance.f1,
            "category_accuracy": self.category_accuracy,
            "per_category": {
                k: {"precision": v.precision, "recall": v.recall, "support": v.tp + v.fn}
                for k, v in self.per_category.items()
            },
            "confusion": self.confusion,
            "missed_important": [
                {"gmail_id": p.gmail_id, "subject": p.subject, "reason": p.reason}
                for p in self.missed_important
            ],
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p95": self.latency_ms_p95,
        }


def score(pairs: list[Pair], *, model_id: str, prompt_version: str) -> Report:
    scored = [p for p in pairs if p.error is None]
    errors = len(pairs) - len(scored)

    importance = Binary()
    for p in scored:
        if p.expected_important and p.predicted_important:
            importance.tp += 1
        elif not p.expected_important and p.predicted_important:
            importance.fp += 1
        elif p.expected_important and not p.predicted_important:
            importance.fn += 1
        else:
            importance.tn += 1

    categories = category_values()
    per_category = {c: Binary() for c in categories}
    confusion = {c: {d: 0 for d in categories} for c in categories}
    correct = 0
    for p in scored:
        if p.expected_category == p.predicted_category:
            correct += 1
        row = confusion.get(p.expected_category)
        if row is not None and p.predicted_category in row:
            confusion[p.expected_category][p.predicted_category] += 1
        for c in categories:
            b = per_category[c]
            if p.expected_category == c and p.predicted_category == c:
                b.tp += 1
            elif p.expected_category != c and p.predicted_category == c:
                b.fp += 1
            elif p.expected_category == c and p.predicted_category != c:
                b.fn += 1

    latencies = sorted(p.latency_ms for p in scored) or [0]
    return Report(
        model_id=model_id,
        prompt_version=prompt_version,
        n=len(scored),
        errors=errors,
        importance=importance,
        category_accuracy=correct / len(scored) if scored else 0.0,
        per_category=per_category,
        confusion=confusion,
        # The rows to read first: every one is an email the user would have
        # wanted today and did not get.
        missed_important=[p for p in scored if p.expected_important and not p.predicted_important],
        latency_ms_p50=latencies[len(latencies) // 2],
        latency_ms_p95=latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)],
    )


def render(report: Report) -> str:
    """Markdown, so two runs can be diffed rather than argued about."""
    lines: list[str] = []
    lines.append(f"# eval: {report.model_id} / {report.prompt_version}")
    lines.append("")
    lines.append(f"{report.n} scored, {report.errors} errored")
    lines.append("")
    lines.append("## importance (primary)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| **recall** | **{report.importance.recall:.3f}** |")
    lines.append(f"| precision | {report.importance.precision:.3f} |")
    lines.append(f"| f1 | {report.importance.f1:.3f} |")
    lines.append(f"| missed important | {len(report.missed_important)} |")
    lines.append(f"| false alarms | {report.importance.fp} |")
    lines.append("")
    lines.append("## category")
    lines.append("")
    lines.append(f"accuracy {report.category_accuracy:.3f}")
    lines.append("")
    lines.append("| category | precision | recall | support |")
    lines.append("| --- | --- | --- | --- |")
    for name, b in report.per_category.items():
        support = b.tp + b.fn
        if not support and not b.fp:
            continue
        lines.append(f"| {name} | {b.precision:.3f} | {b.recall:.3f} | {support} |")
    lines.append("")
    lines.append("## confusion (rows = expected, columns = predicted)")
    lines.append("")
    lines.append(_confusion_table(report.confusion))
    if report.missed_important:
        lines.append("")
        lines.append("## missed important -- read every one of these")
        lines.append("")
        for p in report.missed_important:
            lines.append(f"- `{p.gmail_id}` {p.subject} -- model said: {p.reason}")
    lines.append("")
    lines.append(f"latency p50 {report.latency_ms_p50}ms / p95 {report.latency_ms_p95}ms")
    return "\n".join(lines)


def _confusion_table(confusion: dict[str, dict[str, int]]) -> str:
    """Only rows and columns that actually occur -- a 13x13 grid of zeros
    hides the two categories that are genuinely bleeding into each other."""
    used_rows = [r for r, cols in confusion.items() if any(cols.values())]
    used_cols = sorted({c for r in used_rows for c, n in confusion[r].items() if n})
    if not used_rows:
        return "(no predictions)"
    header = "| expected \\ predicted | " + " | ".join(used_cols) + " |"
    divider = "| --- " * (len(used_cols) + 1) + "|"
    body = [
        "| " + r + " | " + " | ".join(str(confusion[r][c]) for c in used_cols) + " |"
        for r in used_rows
    ]
    return "\n".join([header, divider, *body])

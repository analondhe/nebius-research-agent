"""Research trace: an ordered, structured record of one run.

Every value is redacted before it is stored, so API keys never reach the trace,
its JSONL export or the terminal, even when an error message or web text echoes one.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

REDACTED = "[REDACTED]"
# Key-shaped tokens redacted even when the exact value is unknown.
_KEY_PATTERNS = (
    re.compile(r"tvly-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._~+/=-]{8,}"),
)
_MIN_SECRET_LEN = 6


class Trace:
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets = sorted({s for s in secrets if s and len(s) >= _MIN_SECRET_LEN}, key=len, reverse=True)
        self.events: list[dict[str, Any]] = []

    def record(self, event: str, message: str, *, level: str = "info", **fields: Any) -> None:
        entry = {"step": len(self.events) + 1, "event": event, "level": level, "message": message, **fields}
        self.events.append(self._redact(entry))

    def warn(self, event: str, message: str, **fields: Any) -> None:
        self.record(event, message, level="warning", **fields)

    def of(self, event: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == event]

    def warnings(self) -> list[str]:
        return [e["message"] for e in self.events if e["level"] == "warning"]

    def to_jsonl(self) -> str:
        return "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in self.events)

    def render(self) -> str:
        return "\n".join(
            f"{e['step']:>3}  {e['event']:<15} {'! ' if e['level'] == 'warning' else ''}{e['message']}" for e in self.events
        )

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, REDACTED)
            for pattern in _KEY_PATTERNS:
                value = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + REDACTED, value)
            return value
        if isinstance(value, dict):
            return {k: self._redact(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact(v) for v in value]
        return value

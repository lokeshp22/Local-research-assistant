"""Append-only JSONL run log.

One JSON object per line, one line per run. The orchestrator and the eval
harness are the only writers; the dashboard opens this file read-only.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
_WRITE_LOCK = threading.Lock()


def append_run(path: str | Path, record: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return path


def iter_runs(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield every well-formed record. A truncated final line (process killed
    mid-write) is skipped rather than blowing up the dashboard."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                record.setdefault("_line", lineno)
                yield record


def read_runs(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_runs(path))

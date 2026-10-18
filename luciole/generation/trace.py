"""Traces des réponses : une ligne JSON par question, un fichier par jour.

Rangées dans ``$LUCIOLE_DATA_DIR/traces/`` (défaut ``data/traces/``) : la
question, les recherches (listes, scores, passages cités), les appels au LLM
(tokens, relances), les gardes, le verdict, les modèles et les versions.
Une trace se retrouve par son ``trace_id``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


def traces_dir() -> Path:
    return Path(os.environ.get("LUCIOLE_DATA_DIR") or "data") / "traces"


class JsonlTracer:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else traces_dir()
        self._lock = threading.Lock()

    def path(self) -> Path:
        return self.directory / f"{time.strftime('%Y-%m-%d')}.jsonl"

    def write(self, trace: dict) -> Path:
        path = self.path()
        line = json.dumps(trace, ensure_ascii=False, default=str)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return path

    def find(self, trace_id: str) -> dict | None:
        """La trace ``trace_id``, cherchée des jours récents aux plus anciens."""
        if not self.directory.is_dir():
            return None
        for path in sorted(self.directory.glob("*.jsonl"), reverse=True):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if trace_id in line:
                        trace = json.loads(line)
                        if trace.get("trace_id") == trace_id:
                            return trace
        return None

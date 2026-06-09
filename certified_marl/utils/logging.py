"""Lightweight JSONL logger."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append one JSON object per line."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r") as f:
            return [json.loads(line) for line in f if line.strip()]

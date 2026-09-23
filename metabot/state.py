"""Tiny persistent state between runs (failed attempts and cost history).

GitHub Actions runners are ephemeral; the workflow restores/saves the
``.bot_state`` folder with actions/cache. If the file is missing or corrupt we
simply start again: the state is an optimisation, never a requirement.

It prevents a question that fails deterministically from being retried (and
paid for) on every run until it closes.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(os.getenv("BOT_STATE_PATH", ".bot_state/state.json"))


class RunState:
    def __init__(self, path: Path = DEFAULT_PATH, max_attempts: int = 3) -> None:
        self.path = Path(path)
        self.max_attempts = max_attempts
        self.data: dict = {"attempts": {}, "costs": []}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable state file %s", self.path)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Keep the file small: last 2000 cost records.
            self.data["costs"] = self.data.get("costs", [])[-2000:]
            self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        except OSError:
            logger.warning("Could not save state file %s", self.path)

    def attempts(self, question_id: int | None) -> int:
        if question_id is None:
            return 0
        return int(self.data["attempts"].get(str(question_id), 0))

    def exhausted(self, question_id: int | None) -> bool:
        return self.attempts(question_id) >= self.max_attempts

    def record_failure(self, question_id: int | None) -> None:
        if question_id is None:
            return
        key = str(question_id)
        self.data["attempts"][key] = self.attempts(question_id) + 1

    def record_success(self, question_id: int | None, cost: float, kind: str, profile: str) -> None:
        if question_id is not None:
            self.data["attempts"].pop(str(question_id), None)
        self.data.setdefault("costs", []).append(
            {
                "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "q": question_id,
                "kind": kind,
                "profile": profile,
                "usd": round(float(cost or 0.0), 4),
            }
        )

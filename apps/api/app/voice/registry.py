"""In-process registry of live and recently ended calls.

Moves to Redis in Phase 13 so multiple API instances share call state.
"""

from __future__ import annotations

from collections import deque

from app.voice.session import CallSession, CallSnapshot


class CallRegistry:
    def __init__(self, recent_limit: int = 50) -> None:
        self._active: dict[str, CallSession] = {}
        # Sessions, not snapshots: a call finished while its close() is still
        # flushing STT keeps reporting its final state once that completes.
        self._recent: deque[CallSession] = deque(maxlen=recent_limit)

    def add(self, session: CallSession) -> None:
        self._active[session.call_sid] = session

    def finish(self, call_sid: str) -> None:
        if (session := self._active.pop(call_sid, None)) is not None:
            self._recent.appendleft(session)

    def sessions(self) -> list[CallSession]:
        return list(self._active.values())

    def active(self) -> list[CallSnapshot]:
        return [s.snapshot() for s in self._active.values()]

    def recent(self) -> list[CallSnapshot]:
        return [s.snapshot() for s in self._recent]

    def get(self, call_sid: str) -> CallSnapshot | None:
        session = self._active.get(call_sid) or next(
            (s for s in self._recent if s.call_sid == call_sid), None
        )
        return session.snapshot() if session else None

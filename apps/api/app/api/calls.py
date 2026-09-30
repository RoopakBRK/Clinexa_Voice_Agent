"""Read-only call inspection API (backs the dashboard in Phase 16)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.voice.deps import get_call_registry
from app.voice.registry import CallRegistry
from app.voice.session import CallSnapshot

router = APIRouter(prefix="/api/calls", tags=["calls"])


@router.get("/active")
async def active_calls(registry: CallRegistry = Depends(get_call_registry)) -> list[CallSnapshot]:
    return registry.active()


@router.get("/recent")
async def recent_calls(registry: CallRegistry = Depends(get_call_registry)) -> list[CallSnapshot]:
    return registry.recent()


@router.get("/{call_sid}")
async def get_call(
    call_sid: str, registry: CallRegistry = Depends(get_call_registry)
) -> CallSnapshot:
    if (snapshot := registry.get(call_sid)) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Call not found")
    return snapshot

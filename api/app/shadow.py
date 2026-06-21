"""Shadow deploy logger.

Keeps an in-memory log of Production vs Staging predictions so the
dashboard A/B comparison tile has data to read.

TODO (Phase 2): swap _log for a real INSERT into the `predictions`
Postgres table once Charlie/Ha's schema is available. The ShadowLogEntry
schema already matches the intended columns.

Owner: Amelia.
"""
from __future__ import annotations

from app.schemas import ShadowLogEntry

# In-memory store. Cleared on container restart — acceptable for Phase 2
# development. Replace with DB writes before Phase 3 demo.
_log: list[ShadowLogEntry] = []


def record(
    text: str,
    production_label: str,
    staging_label: str | None,
) -> None:
    """Append one prediction pair to the log."""
    _log.append(ShadowLogEntry(
        text=text,
        production_label=production_label,
        staging_label=staging_label,
        stage="shadow" if staging_label is not None else "production",
    ))


def get_log() -> list[ShadowLogEntry]:
    """Return all logged entries. Used by the dashboard A/B endpoint."""
    return list(_log)
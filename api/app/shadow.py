from __future__ import annotations

from app.schemas import ShadowLogEntry

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
"""Model loading — delegates to ``models.inference`` for registry + pickle paths.

Owner: Amelia.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from models.inference import ModelSet, SentimentModel, load_models, load_production_model


@dataclass
class LoadedModel:
    """Backward-compatible wrapper around a Production ``SentimentModel``."""

    pipeline: Any
    source: str
    model_name: str = "sentiment-baseline"
    model_version: Optional[str] = None
    stage: str = "Production"

    @classmethod
    def from_sentiment(cls, model: SentimentModel) -> "LoadedModel":
        return cls(
            pipeline=model.model,
            source=model.source,
            model_name=model.model_name,
            model_version=model.model_version,
            stage=model.stage,
        )


def load_model() -> Optional[LoadedModel]:
    """Load Production only (legacy entry point)."""
    production = load_production_model()
    if production is None:
        return None
    return LoadedModel.from_sentiment(production)


def load_model_set() -> ModelSet:
    """Load Production + optional Staging shadow candidate."""
    return load_models()

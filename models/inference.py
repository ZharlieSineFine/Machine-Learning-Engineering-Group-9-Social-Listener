"""Shared sentiment inference — used by FastAPI and Airflow batch scoring.

Loads sklearn baselines from MLflow or a local pickle. DistilBERT (pytorch)
is supported when ``torch`` + ``transformers`` are installed; otherwise the
shadow lane is skipped with a warning.

Owner: Amelia (+ Van for model contracts).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from models.baseline_sklearn import LABELS, TunedSentimentPipeline, clean_text

DEFAULT_PICKLE = Path(
    os.getenv(
        "MODEL_PICKLE_PATH",
        str(Path(__file__).resolve().parents[1] / "models" / "artifacts" / "baseline.pkl"),
    )
)
DEFAULT_MODEL_NAME = "sentiment-baseline"
DEFAULT_MODEL_STAGE = "Production"
DEFAULT_SHADOW_NAME = "sentiment-distilbert"
DEFAULT_SHADOW_STAGE = "Staging"


@dataclass
class PredictionResult:
    label: str
    score: Optional[float] = None


@dataclass
class SentimentModel:
    """A loaded classifier plus registry metadata."""

    model: Any
    model_name: str
    model_version: Optional[str]
    stage: str
    source: str  # mlflow | pickle

    @property
    def pipeline(self) -> Any:
        """Backward-compatible alias used by existing API tests."""
        return self.model


def prepare_texts(texts: Sequence[str]) -> list[str]:
    return [clean_text(t) for t in texts]


def predict_labels(model: Any, texts: Sequence[str]) -> list[str]:
    """Return one label per input text."""
    cleaned = prepare_texts(texts)
    preds = model.predict(cleaned)
    return [str(p) for p in preds]


def predict_with_scores(model: Any, texts: Sequence[str]) -> list[PredictionResult]:
    """Predict labels and attach a confidence score when the model supports it."""
    cleaned = prepare_texts(texts)
    labels = predict_labels(model, texts)
    scores: list[Optional[float]] = [None] * len(labels)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(cleaned)
        label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}
        for i, lbl in enumerate(labels):
            idx = label_to_idx.get(lbl)
            if idx is not None and idx < probas.shape[1]:
                scores[i] = float(probas[i, idx])

    return [
        PredictionResult(label=lbl, score=score)
        for lbl, score in zip(labels, scores)
    ]


def _mlflow_version(model_name: str, stage: str) -> Optional[str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        versions = mlflow.tracking.MlflowClient().get_latest_versions(
            model_name, stages=[stage]
        )
        return versions[0].version if versions else None
    except Exception:
        return None


def _load_sklearn_mlflow(model_name: str, stage: str) -> SentimentModel:
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(uri)
    return SentimentModel(
        model=model,
        model_name=model_name,
        model_version=_mlflow_version(model_name, stage),
        stage=stage,
        source="mlflow",
    )


def _load_distilbert_wrapper(model_name: str, stage: str) -> SentimentModel:
    """Wrap a registered DistilBERT checkpoint with thresholded predict()."""
    from transformers import AutoTokenizer

    from models.distilbert_finetune import DEFAULT_NEG_THRESHOLD, TrainConfig, _predict_labels

    import mlflow
    import mlflow.pytorch

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    uri = f"models:/{model_name}/{stage}"
    pytorch_model = mlflow.pytorch.load_model(uri)
    tokenizer = AutoTokenizer.from_pretrained(pytorch_model.config._name_or_path)
    cfg = TrainConfig(neg_threshold=DEFAULT_NEG_THRESHOLD)

    class _DistilbertPredictor:
        def predict(self, texts: Sequence[str]) -> list[str]:
            return _predict_labels(pytorch_model, tokenizer, list(texts), cfg)

    return SentimentModel(
        model=_DistilbertPredictor(),
        model_name=model_name,
        model_version=_mlflow_version(model_name, stage),
        stage=stage,
        source="mlflow",
    )


def load_from_mlflow(model_name: str, stage: str) -> Optional[SentimentModel]:
    """Load a registered model. Returns None if MLflow is not configured."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return None

    if "distilbert" in model_name.lower():
        try:
            return _load_distilbert_wrapper(model_name, stage)
        except ImportError:
            print(
                f"[inference] torch/transformers not installed; "
                f"cannot load shadow model {model_name}/{stage}"
            )
            return None
        except Exception as exc:
            print(f"[inference] could not load {model_name}/{stage}: {exc}")
            return None

    try:
        return _load_sklearn_mlflow(model_name, stage)
    except Exception as exc:
        print(f"[inference] could not load {model_name}/{stage}: {exc}")
        return None


def load_from_pickle(
    path: Path | str = DEFAULT_PICKLE,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    stage: str = DEFAULT_MODEL_STAGE,
) -> SentimentModel:
    with open(path, "rb") as f:
        model = pickle.load(f)
    return SentimentModel(
        model=model,
        model_name=model_name,
        model_version=None,
        stage=stage,
        source="pickle",
    )


def load_production_model() -> Optional[SentimentModel]:
    """Best-effort load of the Production lane."""
    name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    stage = os.getenv("MODEL_STAGE", DEFAULT_MODEL_STAGE)

    via_mlflow = load_from_mlflow(name, stage)
    if via_mlflow is not None:
        return via_mlflow
    if DEFAULT_PICKLE.exists():
        return load_from_pickle(DEFAULT_PICKLE, model_name=name, stage=stage)
    print(f"[inference] No model at {DEFAULT_PICKLE} and MLflow not configured")
    return None


def load_shadow_model() -> Optional[SentimentModel]:
    """Load the Staging shadow candidate.

    Disabled when ``SHADOW_MODEL_NAME`` is set to an empty string.
    """
    raw = os.getenv("SHADOW_MODEL_NAME")
    if raw is not None and not raw.strip():
        return None
    name = raw or DEFAULT_SHADOW_NAME
    stage = os.getenv("SHADOW_MODEL_STAGE", DEFAULT_SHADOW_STAGE)
    return load_from_mlflow(name, stage)


@dataclass
class ModelSet:
    production: Optional[SentimentModel]
    shadow: Optional[SentimentModel] = None


def load_models() -> ModelSet:
    """Load Production (required for serving) and optional Staging shadow."""
    production = load_production_model()
    shadow = load_shadow_model()
    if shadow is not None and production is not None:
        if (
            shadow.model_name == production.model_name
            and shadow.model_version == production.model_version
        ):
            shadow = None
    return ModelSet(production=production, shadow=shadow)

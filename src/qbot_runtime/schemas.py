from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    capability_id: str
    artifact_uri: str
    runtime_version: str
    provider_revision: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    artifact: ModelArtifact
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    values: list[float]
    probabilities: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

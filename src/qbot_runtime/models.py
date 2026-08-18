from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .capabilities import QBOT_REVISION, RUNTIME_VERSION, capabilities
from .schemas import ModelArtifact, PredictionResult, TrainingResult


class ModelRuntime(Protocol):
    capability_id: str
    task: str

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[float], output_uri: str | Path) -> TrainingResult: ...
    def predict_batch(self, features: Sequence[Sequence[float]]) -> PredictionResult: ...
    def save(self, output_uri: str | Path) -> ModelArtifact: ...


class SklearnCompatibleModelRuntime:
    """Stable facade for estimators following the sklearn fit/predict contract."""

    def __init__(self, capability_id: str, task: str, estimator: Any, config: dict[str, Any]) -> None:
        self.capability_id = capability_id
        self.task = task
        self.estimator = estimator
        self.config = dict(config)
        self.feature_count: int | None = None

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[float], output_uri: str | Path) -> TrainingResult:
        matrix = _matrix(features)
        values = np.asarray(targets)
        if matrix.shape[0] != values.shape[0]:
            raise ValueError("features and targets must have the same number of rows")
        if matrix.shape[0] == 0:
            raise ValueError("training data must not be empty")
        self.feature_count = int(matrix.shape[1])
        self.estimator.fit(matrix, values)
        artifact = self.save(output_uri)
        score = float(self.estimator.score(matrix, values))
        return TrainingResult(artifact, {"training_score": score, "sample_count": float(matrix.shape[0])})

    def predict_batch(self, features: Sequence[Sequence[float]]) -> PredictionResult:
        matrix = _matrix(features)
        if self.feature_count is None:
            raise RuntimeError("model must be fitted or loaded before prediction")
        if matrix.shape[1] != self.feature_count:
            raise ValueError(f"expected {self.feature_count} features, received {matrix.shape[1]}")
        values = np.asarray(self.estimator.predict(matrix), dtype=float).tolist()
        probabilities = _positive_probabilities(self.estimator, matrix) if self.task == "classification" else None
        return PredictionResult(values, probabilities, self._metadata())

    def save(self, output_uri: str | Path) -> ModelArtifact:
        if self.feature_count is None:
            raise RuntimeError("model must be fitted before it can be saved")
        path = Path(output_uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "capability_id": self.capability_id,
            "task": self.task,
            "config": self.config,
            "feature_count": self.feature_count,
            "runtime_version": RUNTIME_VERSION,
            "qbot_revision": QBOT_REVISION,
            "estimator": self.estimator,
        }
        path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ModelArtifact(self.capability_id, str(path), RUNTIME_VERSION, QBOT_REVISION, {**self._metadata(), "artifact_sha256": digest})

    def _metadata(self) -> dict[str, Any]:
        return {
            "provider": "qbot",
            "capability_id": self.capability_id,
            "task": self.task,
            "feature_count": self.feature_count,
            "runtime_version": RUNTIME_VERSION,
            "qbot_revision": QBOT_REVISION,
        }


def create_model(capability_id: str, config: dict[str, Any] | None = None) -> SklearnCompatibleModelRuntime:
    selected = next((item for item in capabilities().capabilities if item.id == capability_id), None)
    if selected is None or selected.kind != "model":
        raise ValueError(f"Unknown Qbot model capability: {capability_id}")
    if not selected.available:
        raise RuntimeError(selected.unavailable_reason or f"{capability_id} is unavailable")
    options = dict(config or {})
    task = str(options.pop("task", selected.tasks[0]))
    if task not in selected.tasks:
        raise ValueError(f"{capability_id} does not support task {task}")
    random_state = int(options.pop("random_state", 42))
    estimator = _create_estimator(capability_id, task, random_state, options)
    return SklearnCompatibleModelRuntime(capability_id, task, estimator, {"task": task, "random_state": random_state, **options})


def load_model(artifact_uri: str | Path, *, expected_sha256: str | None = None) -> SklearnCompatibleModelRuntime:
    path = Path(artifact_uri)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Qbot model artifact SHA256 mismatch")
    payload = pickle.loads(raw)
    required = {"format_version", "capability_id", "task", "config", "feature_count", "estimator"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Unsupported Qbot model artifact")
    if payload["format_version"] != 1:
        raise ValueError(f"Unsupported Qbot artifact format: {payload['format_version']}")
    runtime = SklearnCompatibleModelRuntime(str(payload["capability_id"]), str(payload["task"]), payload["estimator"], dict(payload["config"]))
    runtime.feature_count = int(payload["feature_count"])
    return runtime


def _matrix(features: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional numeric matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    return matrix


def _positive_probabilities(estimator: Any, matrix: np.ndarray) -> list[float] | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    probabilities = np.asarray(estimator.predict_proba(matrix), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return None
    return probabilities[:, -1].tolist()


def _create_estimator(capability_id: str, task: str, random_state: int, options: dict[str, Any]) -> Any:
    if capability_id == "qbot.linear_regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**options)
    if capability_id == "qbot.logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=random_state, **options)
    if capability_id == "qbot.random_forest":
        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=random_state, **options)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=random_state, **options)
    if capability_id == "qbot.gradient_boosting":
        if task == "classification":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=random_state, **options)
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=random_state, **options)
    if capability_id == "qbot.lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        model = LGBMClassifier if task == "classification" else LGBMRegressor
        return model(random_state=random_state, **options)
    if capability_id == "qbot.xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        model = XGBClassifier if task == "classification" else XGBRegressor
        return model(random_state=random_state, **options)
    if capability_id == "qbot.catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        model = CatBoostClassifier if task == "classification" else CatBoostRegressor
        return model(random_seed=random_state, verbose=False, **options)
    raise ValueError(f"Unsupported Qbot model capability: {capability_id}")

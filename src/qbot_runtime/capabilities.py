from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any


RUNTIME_VERSION = "0.1.0"
QBOT_REVISION = "f0425ae4ae8bd02b79656b8f7039f4cd6874095e"


@dataclass(frozen=True, slots=True)
class Capability:
    id: str
    kind: str
    markets: tuple[str, ...]
    tasks: tuple[str, ...]
    dependency: str
    available: bool
    unavailable_reason: str | None = None
    supports_fit: bool = False
    supports_batch_predict: bool = False

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["markets"] = list(self.markets)
        result["tasks"] = list(self.tasks)
        return result


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    runtime_version: str
    qbot_revision: str
    capabilities: tuple[Capability, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime_version": self.runtime_version,
            "qbot_revision": self.qbot_revision,
            "capabilities": [item.as_dict() for item in self.capabilities],
        }


def _model(capability_id: str, dependency: str, tasks: tuple[str, ...]) -> Capability:
    available = find_spec(dependency) is not None
    return Capability(
        id=capability_id,
        kind="model",
        markets=("a_share", "crypto"),
        tasks=tasks,
        dependency=dependency,
        available=available,
        unavailable_reason=None if available else f"Install the optional dependency providing {dependency}",
        supports_fit=True,
        supports_batch_predict=True,
    )


def capabilities() -> CapabilityManifest:
    """Return only facade capabilities with implemented stable contracts."""
    items = (
        _model("qbot.linear_regression", "sklearn", ("regression",)),
        _model("qbot.logistic_regression", "sklearn", ("classification",)),
        _model("qbot.random_forest", "sklearn", ("regression", "classification")),
        _model("qbot.gradient_boosting", "sklearn", ("regression", "classification")),
        _model("qbot.lightgbm", "lightgbm", ("regression", "classification")),
        _model("qbot.xgboost", "xgboost", ("regression", "classification")),
        _model("qbot.catboost", "catboost", ("regression", "classification")),
    )
    return CapabilityManifest(RUNTIME_VERSION, QBOT_REVISION, items)

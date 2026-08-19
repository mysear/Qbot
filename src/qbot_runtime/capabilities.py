from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any


RUNTIME_VERSION = "0.5.0"
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
    config_schema: dict[str, Any] | None = None
    display_name: str = ""
    timeframes: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()
    supports_evaluate: bool = False
    devices: tuple[str, ...] = ("cpu",)
    maturity: str = "stable"

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["markets"] = list(self.markets)
        result["tasks"] = list(self.tasks)
        result["timeframes"] = list(self.timeframes)
        result["horizons"] = list(self.horizons)
        result["devices"] = list(self.devices)
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
    available, reason = _dependency_state(dependency)
    return Capability(
        id=capability_id,
        kind="model",
        markets=("a_share", "crypto"),
        tasks=tasks,
        dependency=dependency,
        available=available,
        unavailable_reason=reason,
        supports_fit=True,
        supports_batch_predict=True,
        supports_evaluate=True,
        display_name=capability_id.removeprefix("qbot.").replace("_", " ").title(),
        timeframes=("1m", "5m", "15m", "1h", "4h", "1d"),
        config_schema={"type": "object", "additionalProperties": True},
    )


def _capability(capability_id: str, kind: str, dependency: str = "", *, markets: tuple[str, ...] = ("a_share", "crypto"), tasks: tuple[str, ...] = (), maturity: str = "stable", config_schema: dict[str, Any] | None = None) -> Capability:
    available, reason = _dependency_state(dependency)
    return Capability(
        capability_id, kind, markets, tasks, dependency, available,
        reason, config_schema=config_schema, display_name=capability_id.removeprefix("qbot.").replace("_", " ").title(), maturity=maturity,
    )


def _dependency_state(dependency: str) -> tuple[bool, str | None]:
    if not dependency:
        return True, None
    if find_spec(dependency) is None:
        return False, f"Install the optional dependency providing {dependency}"
    try:
        import_module(dependency)
    except Exception as exc:
        return False, f"Cannot import optional dependency {dependency}: {exc}"
    return True, None


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
        _model("qbot.mlp", "torch", ("regression", "classification")),
        _model("qbot.lstm", "torch", ("regression", "classification")),
        _model("qbot.gru", "torch", ("regression", "classification")),
        _model("qbot.transformer", "torch", ("regression", "classification")),
        _model("qbot.tft", "torch", ("regression", "classification")),
        _capability("qbot.ta", "feature"),
        _capability("qbot.ma", "strategy", markets=("a_share","crypto","fund","futures")),
        _capability("qbot.momentum", "strategy", markets=("a_share","crypto","fund","futures")),
        _capability("qbot.multi_factor", "strategy", markets=("a_share","crypto","fund","futures")),
        _capability("qbot.q_learning", "strategy", markets=("a_share","crypto","fund","futures"), config_schema={"type":"object","properties":{"episodes":{"type":"integer","minimum":1},"random_state":{"type":"integer"}},"additionalProperties":True}),
        _capability("qbot.factor_mining", "factor_workflow", markets=("a_share","crypto","fund","futures"), config_schema={"type":"object","properties":{"windows":{"type":"array","items":{"type":"integer","minimum":2}},"horizon":{"type":"integer","minimum":1}},"additionalProperties":False}),
        _capability("qbot.vector_backtest", "backtest", markets=("a_share","crypto","fund","futures")),
        _capability("qbot.backtrader", "backtest", "backtrader", markets=("a_share","crypto","fund","futures")),
        _capability("qbot.binance", "data", "httpx", markets=("crypto",)),
        _capability("qbot.akshare", "data", "akshare", markets=("a_share",)),
        _capability("qbot.yfinance", "data", "yfinance"),
        _capability("qbot.fund_yfinance", "data", "yfinance", markets=("fund",)),
        _capability("qbot.futures_yfinance", "data", "yfinance", markets=("futures",)),
        _capability("qbot.efinance", "data", "efinance", markets=("a_share",)),
        _capability("qbot.paper", "execution"),
    )
    return CapabilityManifest(RUNTIME_VERSION, QBOT_REVISION, items)

from __future__ import annotations

from typing import Any, Protocol, Sequence

import pandas as pd


class StrategyRuntime(Protocol):
    capability_id: str
    def generate(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]: ...


class SignalStrategyRuntime:
    def __init__(self, capability_id: str, config: dict[str, Any] | None = None) -> None:
        self.capability_id = capability_id
        self.config = dict(config or {})

    def generate(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        frame = pd.DataFrame(rows).copy()
        if "close" not in frame or frame.empty:
            raise ValueError("Strategy rows require non-empty close values")
        close = pd.to_numeric(frame.close, errors="raise")
        if self.capability_id == "qbot.ma":
            fast = close.rolling(int(self.config.get("fast", 5)), min_periods=1).mean()
            slow = close.rolling(int(self.config.get("slow", 20)), min_periods=1).mean()
            score = (fast / slow.replace(0, float("nan")) - 1).fillna(0)
        elif self.capability_id == "qbot.momentum":
            score = close.pct_change(int(self.config.get("lookback", 20))).fillna(0)
        elif self.capability_id == "qbot.multi_factor":
            momentum = close.pct_change(int(self.config.get("lookback", 20))).fillna(0)
            volatility = close.pct_change().rolling(int(self.config.get("volatility_window", 20)), min_periods=2).std().fillna(0)
            score = momentum - float(self.config.get("risk_weight", 1.0)) * volatility
        else:
            raise ValueError(f"Unsupported Qbot strategy capability: {self.capability_id}")
        threshold = float(self.config.get("threshold", 0.0))
        signals = score.map(lambda value: 1 if value > threshold else -1 if value < -threshold else 0)
        return [{"index": int(index), "signal": int(signal), "score": float(value), "target_weight": float(signal)} for index, (signal, value) in enumerate(zip(signals, score, strict=True))]


def create_strategy(capability_id: str, config: dict[str, Any] | None = None) -> SignalStrategyRuntime:
    if capability_id == "qbot.q_learning":
        from .ai_strategies import QLearningStrategyRuntime

        return QLearningStrategyRuntime(config)
    if capability_id not in {"qbot.ma", "qbot.momentum", "qbot.multi_factor"}:
        raise ValueError(f"Unknown Qbot strategy capability: {capability_id}")
    return SignalStrategyRuntime(capability_id, config)

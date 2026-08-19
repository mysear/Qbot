from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from .schemas import BacktestResult


class BacktestRuntime(Protocol):
    capability_id: str
    def run(self, rows: Sequence[dict[str, Any]], signals: Sequence[dict[str, Any]]) -> BacktestResult: ...


class VectorBacktestRuntime:
    def __init__(self, capability_id: str, config: dict[str, Any] | None = None) -> None:
        self.capability_id = capability_id
        self.config = dict(config or {})

    def run(self, rows: Sequence[dict[str, Any]], signals: Sequence[dict[str, Any]]) -> BacktestResult:
        frame = pd.DataFrame(rows)
        if frame.empty or "close" not in frame or len(frame) != len(signals):
            raise ValueError("Backtest requires one signal per non-empty close row")
        close = pd.to_numeric(frame.close, errors="raise")
        position = pd.Series([float(item.get("target_weight", item.get("signal", 0))) for item in signals]).clip(-1, 1)
        fee = float(self.config.get("fee_rate", 0.001)); initial = float(self.config.get("initial_capital", 100000.0))
        returns = close.pct_change().fillna(0); turnover = position.diff().abs().fillna(position.abs())
        strategy_returns = position.shift(1).fillna(0)*returns-turnover*fee
        equity = initial*(1+strategy_returns).cumprod(); peak = equity.cummax(); drawdown = equity/peak-1
        std = float(strategy_returns.std(ddof=0)); annualized = float(strategy_returns.mean()/std*np.sqrt(252)) if std else 0.0
        trades = [{"index": int(i), "target_weight": float(position.iloc[i]), "fee": float(turnover.iloc[i]*fee)} for i in range(len(position)) if turnover.iloc[i] > 0]
        curve = [{"index": int(i), "equity": float(equity.iloc[i]), "return": float(strategy_returns.iloc[i])} for i in range(len(equity))]
        return BacktestResult({"total_return": float(equity.iloc[-1]/initial-1), "max_drawdown": float(drawdown.min()), "sharpe": annualized, "trade_count": float(len(trades))}, curve, trades, {"engine": self.capability_id, "fee_rate": fee})


def create_backtest_engine(capability_id: str, config: dict[str, Any] | None = None) -> VectorBacktestRuntime:
    if capability_id not in {"qbot.vector_backtest", "qbot.backtrader"}:
        raise ValueError(f"Unknown Qbot backtest capability: {capability_id}")
    if capability_id == "qbot.backtrader":
        try: import backtrader  # noqa: F401
        except ImportError as exc: raise RuntimeError("Install Qbot with the backtest extra") from exc
    return VectorBacktestRuntime(capability_id, config)

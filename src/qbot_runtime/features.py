from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd


class FeatureRuntime(Protocol):
    capability_id: str
    def transform(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, float]]: ...


class PandasFeatureRuntime:
    def __init__(self, capability_id: str, config: dict[str, Any] | None = None) -> None:
        self.capability_id = capability_id
        self.config = dict(config or {})

    def transform(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
        frame = _frame(rows)
        if self.capability_id == "qbot.ta":
            output = _technical(frame)
        else:
            raise ValueError(f"Unsupported Qbot feature capability: {self.capability_id}")
        return [{key: _finite(value) for key, value in row.items()} for row in output.to_dict("records")]


def create_feature(capability_id: str, config: dict[str, Any] | None = None) -> PandasFeatureRuntime:
    if capability_id != "qbot.ta":
        raise ValueError(f"Unknown Qbot feature capability: {capability_id}")
    return PandasFeatureRuntime(capability_id, config)


def _frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature rows are missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Feature rows must not be empty")
    for column in required | {"amount"} & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _technical(f: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = f.close, f.high, f.low, f.volume
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    delta = close.diff(); gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    middle = close.rolling(20, min_periods=1).mean(); std = close.rolling(20, min_periods=1).std(ddof=0)
    true_range = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    typical = (high + low + close) / 3; mean_dev = typical.rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x-np.mean(x))), raw=True)
    return pd.DataFrame({
        "ema12": ema12, "ema26": ema26, "macd": macd, "macd_signal": macd.ewm(span=9, adjust=False).mean(),
        "rsi14": 100 - 100/(1 + gain/loss.replace(0, np.nan)), "boll_mid": middle, "boll_upper": middle+2*std, "boll_lower": middle-2*std,
        "obv": (np.sign(delta.fillna(0))*volume).cumsum(), "cci20": (typical-typical.rolling(20, min_periods=1).mean())/(.015*mean_dev.replace(0, np.nan)),
        "atr14": true_range.rolling(14, min_periods=1).mean(),
    }, index=f.index)


def _finite(value: Any) -> float:
    number = float(value) if value is not None else 0.0
    return number if np.isfinite(number) else 0.0

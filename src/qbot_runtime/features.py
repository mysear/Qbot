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
        elif self.capability_id == "qbot.alpha101":
            output = _alpha101(frame)
        elif self.capability_id == "qbot.alpha191":
            output = _alpha191(frame)
        else:
            raise ValueError(f"Unsupported Qbot feature capability: {self.capability_id}")
        return [{key: _finite(value) for key, value in row.items()} for row in output.to_dict("records")]


def create_feature(capability_id: str, config: dict[str, Any] | None = None) -> PandasFeatureRuntime:
    if capability_id not in {"qbot.ta", "qbot.alpha101", "qbot.alpha191"}:
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


def _alpha101(f: pd.DataFrame) -> pd.DataFrame:
    returns = f.close.pct_change(); vwap = f.get("amount", f.close*f.volume)/f.volume.replace(0, np.nan)
    return pd.DataFrame({
        "alpha001": returns.rolling(20, min_periods=2).std().rank(pct=True),
        "alpha002": -np.log(f.volume.replace(0, np.nan)).diff(2).rolling(6, min_periods=2).corr((f.close-f.open)/f.open.replace(0, np.nan)),
        "alpha003": -f.open.rank(pct=True).rolling(10, min_periods=2).corr(f.volume.rank(pct=True)),
        "alpha004": -f.low.rank(pct=True).rolling(9, min_periods=1).min(),
        "alpha005": (f.open-vwap.rolling(10, min_periods=1).mean()).rank(pct=True)*-(f.close-vwap).abs().rank(pct=True),
        "alpha006": -f.open.rolling(10, min_periods=2).corr(f.volume),
        "alpha007": -returns.diff(7).abs().rank(pct=True)*np.sign(returns.diff(7)),
        "alpha008": -(f.open.rolling(5, min_periods=1).sum()*returns.rolling(5, min_periods=1).sum()).diff(10).rank(pct=True),
        "alpha009": np.where(returns.rolling(5, min_periods=1).min()>0, returns, np.where(returns.rolling(5, min_periods=1).max()<0, returns, -returns)),
        "alpha010": pd.Series(np.where(returns.rolling(4, min_periods=1).min()>0, returns, -returns), index=f.index).rank(pct=True),
    }, index=f.index)


def _alpha191(f: pd.DataFrame) -> pd.DataFrame:
    ret = f.close.pct_change(); adv20 = f.volume.rolling(20, min_periods=1).mean(); typical = (f.high+f.low+f.close)/3
    return pd.DataFrame({
        "alpha001": -np.log(f.volume.replace(0, np.nan)).diff().rolling(6, min_periods=2).corr((f.close-f.open)/f.open.replace(0, np.nan)),
        "alpha002": -((f.close-f.low)-(f.high-f.close))/(f.high-f.low).replace(0, np.nan),
        "alpha003": f.close.diff().where(f.close.diff()==0, f.close-np.where(f.close.diff()>0, np.minimum(f.low,f.close.shift()), np.maximum(f.high,f.close.shift()))).rolling(6, min_periods=1).sum(),
        "alpha004": np.where(f.close.rolling(8, min_periods=1).mean()+f.close.rolling(8, min_periods=1).std()<f.close.rolling(2, min_periods=1).mean(), -1, 1),
        "alpha005": -f.volume.rolling(5, min_periods=2).corr(f.high.rank(pct=True)).rolling(3, min_periods=1).max(),
        "alpha006": -np.sign((f.open*.85+f.high*.15).diff(4)).rank(pct=True),
        "alpha007": (typical-f.close).rolling(3, min_periods=1).max().rank(pct=True)+(typical-f.close).rolling(3, min_periods=1).min().rank(pct=True)*f.volume.diff(3).rank(pct=True),
        "alpha008": -((f.high+f.low)/2*.2+typical*.8).diff(4).rank(pct=True),
        "alpha009": ((f.high+f.low)/2-(f.high.shift()+f.low.shift())/2)*(f.high-f.low)/f.volume.replace(0, np.nan),
        "alpha010": ret.rolling(20, min_periods=2).std()*f.close + (f.volume/adv20.replace(0, np.nan)-1),
    }, index=f.index)


def _finite(value: Any) -> float:
    number = float(value) if value is not None else 0.0
    return number if np.isfinite(number) else 0.0

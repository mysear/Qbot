from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class DataProvider(Protocol):
    capability_id: str
    def history(self, symbol: str, start: str, end: str, interval: str = "1d") -> list[dict[str, Any]]: ...


class UpstreamDataProvider:
    def __init__(self, capability_id: str, config: dict[str, Any] | None = None) -> None:
        self.capability_id = capability_id
        self.config = dict(config or {})

    def history(self, symbol: str, start: str, end: str, interval: str = "1d") -> list[dict[str, Any]]:
        if self.capability_id == "qbot.yfinance":
            import yfinance as yf
            frame = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        elif self.capability_id == "qbot.akshare":
            import akshare as ak
            frame = ak.stock_zh_a_hist(symbol=symbol, start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust=str(self.config.get("adjust", "qfq")))
            frame = frame.rename(columns={"日期":"date", "开盘":"open", "最高":"high", "最低":"low", "收盘":"close", "成交量":"volume", "成交额":"amount"})
        elif self.capability_id == "qbot.efinance":
            import efinance as ef
            frame = ef.stock.get_quote_history(symbol, beg=start.replace("-", ""), end=end.replace("-", ""))
            frame = frame.rename(columns={"日期":"date", "开盘":"open", "最高":"high", "最低":"low", "收盘":"close", "成交量":"volume", "成交额":"amount"})
        else:
            raise ValueError(f"Unsupported Qbot data provider: {self.capability_id}")
        return _normalize(frame)


def create_data_provider(capability_id: str, config: dict[str, Any] | None = None) -> UpstreamDataProvider:
    if capability_id not in {"qbot.yfinance", "qbot.akshare", "qbot.efinance"}:
        raise ValueError(f"Unknown Qbot data provider: {capability_id}")
    return UpstreamDataProvider(capability_id, config)


def _normalize(frame: pd.DataFrame) -> list[dict[str, Any]]:
    value = frame.reset_index()
    value.columns = [str(item).lower() if not isinstance(item, tuple) else str(item[0]).lower() for item in value.columns]
    if "date" not in value and "index" in value: value = value.rename(columns={"index":"date"})
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = set(required)-set(value.columns)
    if missing: raise ValueError(f"Provider response is missing columns: {', '.join(sorted(missing))}")
    result=[]
    for row in value.to_dict("records"):
        result.append({"date": pd.Timestamp(row["date"]).date().isoformat(), **{key: float(row[key]) for key in required[1:]}, "amount": float(row.get("amount", 0))})
    return result

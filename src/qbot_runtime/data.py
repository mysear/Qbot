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
        if self.capability_id == "qbot.binance":
            start_time = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
            end_time = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
            return self.klines(symbol, interval, start_time=start_time, end_time=end_time)
        if self.capability_id in {"qbot.yfinance", "qbot.fund_yfinance", "qbot.futures_yfinance"}:
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

    def klines(self, symbol: str, interval: str, limit: int = 100, start_time: int | None = None, end_time: int | None = None) -> list[dict[str, Any]]:
        if self.capability_id != "qbot.binance":
            raise ValueError(f"Kline pagination is unsupported by {self.capability_id}")
        import httpx
        client = self.config.get("http_client") or httpx.Client(timeout=float(self.config.get("timeout", 10.0)))
        base_url = str(self.config.get("base_url", "https://api.binance.com")).rstrip("/")
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": int(limit)}
        if start_time is not None: params["startTime"] = int(start_time)
        if end_time is not None: params["endTime"] = int(end_time)
        last_error: Exception | None = None
        for _ in range(int(self.config.get("max_retries", 2))):
            try:
                response = client.get(f"{base_url}/api/v3/klines", params=params)
                response.raise_for_status()
                return [{"open_time": int(item[0]), "open": float(item[1]), "high": float(item[2]), "low": float(item[3]), "close": float(item[4]), "volume": float(item[5]), "amount": float(item[7]), "trade_count": int(item[8]), "taker_buy_volume": float(item[9]), "taker_buy_amount": float(item[10])} for item in response.json()]
            except Exception as exc:
                last_error = exc
        raise last_error if last_error is not None else RuntimeError("Binance kline request failed")


def create_data_provider(capability_id: str, config: dict[str, Any] | None = None) -> UpstreamDataProvider:
    if capability_id not in {"qbot.binance", "qbot.yfinance", "qbot.fund_yfinance", "qbot.futures_yfinance", "qbot.akshare", "qbot.efinance"}:
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

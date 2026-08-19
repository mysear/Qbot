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

    def list_instruments(self, market: str = "a-share") -> list[dict[str, Any]]:
        ak = self._akshare()
        frame = self._call(ak.stock_info_a_code_name)
        return [
            {
                "symbol": _a_share_symbol(str(row["code"])),
                "name": str(row.get("name", "")),
                "exchange": _a_share_exchange(str(row["code"])),
                "is_st": "ST" in str(row.get("name", "")).upper(),
                "status": "listed",
            }
            for row in frame.to_dict("records")
        ]

    def trading_calendar(self, exchange: str, start: str, end: str) -> list[dict[str, Any]]:
        ak = self._akshare()
        frame = self._call(ak.tool_trade_date_hist_sina)
        dates = sorted(str(value)[:10] for value in frame.iloc[:, 0].tolist() if start <= str(value)[:10] <= end)
        return [{"exchange": exchange, "date": value, "is_open": True, "previous_date": dates[index - 1] if index else None, "next_date": dates[index + 1] if index + 1 < len(dates) else None} for index, value in enumerate(dates)]

    def universe_members(self, universe_id: str, as_of: str) -> list[dict[str, Any]]:
        index = {"HS300": "000300", "ZZ500": "000905", "ZZ1000": "000852"}.get(universe_id)
        if index is None:
            return []
        frame = self._call(self._akshare().index_stock_cons, symbol=index)
        code_key = "品种代码" if "品种代码" in frame.columns else "成分券代码"
        return [{"universe_id": universe_id, "symbol": _a_share_symbol(str(row[code_key]).zfill(6)), "effective_from": as_of} for row in frame.to_dict("records")]

    def universe_history(self, universe_id: str, start: str, end: str) -> list[dict[str, Any]]:
        self._akshare()
        raise ValueError("AkShare does not provide authoritative historical index membership; use Tushare or Qlib")

    def suspensions(self, symbols: list[str], start: str, end: str) -> list[dict[str, Any]]:
        from datetime import date, timedelta
        frame = self._call(self._akshare().stock_tfp_em)
        selected = {symbol.split(".")[0]: symbol for symbol in symbols}
        result: list[dict[str, Any]] = []
        for row in frame.to_dict("records"):
            code = str(row.get("代码", "")).zfill(6)
            first = str(row.get("停牌时间", ""))[:10]
            last = str(row.get("停牌截止时间", first))[:10]
            if code not in selected or not first or first > end or last < start:
                continue
            cursor, boundary = max(date.fromisoformat(first), date.fromisoformat(start)), min(date.fromisoformat(last), date.fromisoformat(end))
            while cursor <= boundary:
                result.append({"symbol": selected[code], "date": cursor.isoformat(), "reason": str(row.get("停牌原因", ""))})
                cursor += timedelta(days=1)
        return result

    def instrument_status_history(self, symbols: list[str], start: str, end: str) -> list[dict[str, Any]]:
        selected = set(symbols)
        return [{"symbol": item["symbol"], "effective_from": start, "effective_to": end, "is_st": item["is_st"], "status": item["status"]} for item in self.list_instruments() if item["symbol"] in selected]

    def _akshare(self) -> Any:
        if self.capability_id != "qbot.akshare":
            raise ValueError(f"A-share metadata is unsupported by {self.capability_id}")
        if client := self.config.get("client"):
            return client
        import akshare as ak
        return ak

    def _call(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        import time
        last_error: Exception | None = None
        for attempt in range(int(self.config.get("max_retries", 3))):
            try:
                return operation(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < int(self.config.get("max_retries", 3)):
                    time.sleep(float(self.config.get("retry_delay", 1.0)) * (attempt + 1))
        raise RuntimeError(f"{self.capability_id} request failed: {last_error}") from last_error


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


def _a_share_exchange(code: str) -> str:
    if code.startswith(("4", "8")):
        return "BJ"
    return "SH" if code.startswith(("6", "9")) else "SZ"


def _a_share_symbol(code: str) -> str:
    return f"{code}.{_a_share_exchange(code)}"

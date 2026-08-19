from __future__ import annotations

import pytest

from qbot_runtime import create_backtest_engine, create_data_provider, create_execution_client, create_factor_workflow, create_feature, create_strategy


def bars(count: int = 30) -> list[dict[str, float]]:
    return [{"open": 10+i*.1, "high": 10.2+i*.1, "low": 9.8+i*.1, "close": 10.1+i*.1, "volume": 1000+i*10, "amount": (10.1+i*.1)*(1000+i*10)} for i in range(count)]


def test_feature_facade_is_finite() -> None:
    output = create_feature("qbot.ta").transform(bars())
    assert len(output) == 30
    assert output[-1]
    assert all(value == value and abs(value) != float("inf") for row in output for value in row.values())


def test_unimplemented_alpha_libraries_are_not_advertised() -> None:
    for capability in ("qbot.alpha101", "qbot.alpha191"):
        with pytest.raises(ValueError, match="Unknown Qbot feature capability"):
            create_feature(capability)


@pytest.mark.parametrize("capability", ["qbot.ma", "qbot.momentum", "qbot.multi_factor"])
def test_strategy_facades_return_standard_signals(capability: str) -> None:
    signals = create_strategy(capability).generate(bars())
    assert len(signals) == 30
    assert all(item["signal"] in {-1, 0, 1} and -1 <= item["target_weight"] <= 1 for item in signals)


def test_vector_backtest_returns_standard_report() -> None:
    rows = bars(); signals = create_strategy("qbot.ma").generate(rows)
    report = create_backtest_engine("qbot.vector_backtest").run(rows, signals)
    assert set(report.metrics) >= {"total_return", "max_drawdown", "sharpe", "trade_count"}
    assert len(report.equity_curve) == len(rows)


def test_backtrader_facade_uses_backtrader_engine() -> None:
    pytest.importorskip("backtrader")
    rows = bars(); signals = create_strategy("qbot.ma").generate(rows)
    report = create_backtest_engine("qbot.backtrader").run(rows, signals)
    assert report.config["engine"] == "qbot.backtrader"


def test_paper_execution_is_idempotent() -> None:
    client = create_execution_client("qbot.paper")
    intent = {"symbol":"600001.SH", "side":"buy", "quantity":100, "idempotency_key":"task-1"}
    assert client.submit(intent).order_id == client.submit(intent).order_id


def test_q_learning_round_trip_is_reproducible(tmp_path) -> None:
    runtime = create_strategy("qbot.q_learning", {"episodes": 10, "random_state": 7})
    runtime.fit(bars())
    before = runtime.generate(bars())
    artifact = runtime.save(tmp_path / "q-learning.json")
    loaded = type(runtime).load(artifact["artifact_uri"], artifact["artifact_sha256"])
    assert loaded.generate(bars()) == before


def test_factor_mining_returns_ranked_candidates() -> None:
    result = create_factor_workflow("qbot.factor_mining", {"windows": [3, 5]}).run(bars())
    assert len(result["factors"]) == 6
    assert {"factor_id", "rank_ic", "sample_count"} <= result["factors"][0].keys()


def test_factor_mining_rejects_invalid_windows() -> None:
    with pytest.raises(ValueError, match="windows"):
        create_factor_workflow("qbot.factor_mining", {"windows": [1]}).run(bars())


def test_binance_data_facade_normalizes_paginated_klines() -> None:
    import httpx
    payload = [[1, "100", "101", "99", "100.5", "10", 2, "1000", 5, "6", "600", "0"]]
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    provider = create_data_provider("qbot.binance", {"http_client": client})
    rows = provider.klines("BTCUSDT", "15m", limit=1)
    assert rows == [{"open_time": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0, "amount": 1000.0, "trade_count": 5, "taker_buy_volume": 6.0, "taker_buy_amount": 600.0}]

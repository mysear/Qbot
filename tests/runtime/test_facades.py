from __future__ import annotations

import pytest

from qbot_runtime import create_backtest_engine, create_execution_client, create_feature, create_strategy


def bars(count: int = 30) -> list[dict[str, float]]:
    return [{"open": 10+i*.1, "high": 10.2+i*.1, "low": 9.8+i*.1, "close": 10.1+i*.1, "volume": 1000+i*10, "amount": (10.1+i*.1)*(1000+i*10)} for i in range(count)]


@pytest.mark.parametrize("capability", ["qbot.ta", "qbot.alpha101", "qbot.alpha191"])
def test_feature_facades_are_finite(capability: str) -> None:
    output = create_feature(capability).transform(bars())
    assert len(output) == 30
    assert output[-1]
    assert all(value == value and abs(value) != float("inf") for row in output for value in row.values())


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


def test_paper_execution_is_idempotent() -> None:
    client = create_execution_client("qbot.paper")
    intent = {"symbol":"600001.SH", "side":"buy", "quantity":100, "idempotency_key":"task-1"}
    assert client.submit(intent).order_id == client.submit(intent).order_id

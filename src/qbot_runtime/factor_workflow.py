from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


class FactorMiningRuntime:
    capability_id = "qbot.factor_mining"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def run(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        frame = pd.DataFrame(rows)
        close = pd.to_numeric(frame["close"], errors="raise")
        volume = pd.to_numeric(frame["volume"], errors="raise")
        horizon = int(self.config.get("horizon", 1))
        windows = [int(window) for window in self.config.get("windows", [5, 10, 20])]
        if horizon < 1:
            raise ValueError("Factor mining horizon must be at least one")
        if not windows or any(window < 2 for window in windows):
            raise ValueError("Factor mining windows must contain values of at least two")

        future = close.shift(-horizon) / close - 1
        candidates = {}
        for window in windows:
            candidates[f"momentum_{window}"] = close.pct_change(window)
            candidates[f"volatility_{window}"] = close.pct_change().rolling(window).std()
            candidates[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean() - 1
        scores = []
        for name, values in candidates.items():
            valid = pd.concat([values, future], axis=1).dropna()
            score = float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")) if len(valid) > 2 else 0.0
            if not np.isfinite(score):
                score = 0.0
            scores.append({"factor_id": name, "rank_ic": score, "sample_count": len(valid)})
        scores.sort(key=lambda item: abs(item["rank_ic"]), reverse=True)
        minimum = float(self.config.get("minimum_abs_rank_ic", 0.0))
        return {"factors": scores, "selected": [item for item in scores if abs(item["rank_ic"]) >= minimum]}


def create_factor_workflow(capability_id: str, config: dict[str, Any] | None = None) -> FactorMiningRuntime:
    if capability_id != "qbot.factor_mining":
        raise ValueError(f"Unknown Qbot factor workflow: {capability_id}")
    return FactorMiningRuntime(config)

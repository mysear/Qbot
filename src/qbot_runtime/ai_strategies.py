from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


class QLearningStrategyRuntime:
    """Deterministic tabular Q-learning strategy without data-source side effects."""

    capability_id = "qbot.q_learning"
    actions = (-1, 0, 1)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.q_table: dict[int, list[float]] = {}

    def fit(self, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
        close = pd.to_numeric(pd.DataFrame(rows)["close"], errors="raise").to_numpy(float)
        if len(close) < 3:
            raise ValueError("Q-learning requires at least three close rows")
        if np.any(close[:-1] == 0):
            raise ValueError("Q-learning close prices must be non-zero")
        episodes = int(self.config.get("episodes", 100))
        if episodes < 1:
            raise ValueError("Q-learning episodes must be at least one")

        returns = np.diff(close) / close[:-1]
        bins = np.quantile(returns, [1 / 3, 2 / 3])
        states = np.digitize(returns, bins).astype(int)
        rng = np.random.default_rng(int(self.config.get("random_state", 42)))
        alpha = float(self.config.get("learning_rate", 0.1))
        gamma = float(self.config.get("discount_factor", 0.9))
        epsilon = float(self.config.get("epsilon", 0.1))
        for _ in range(episodes):
            for index, state in enumerate(states[:-1]):
                values = self.q_table.setdefault(int(state), [0.0, 0.0, 0.0])
                action = int(rng.integers(0, 3)) if rng.random() < epsilon else int(np.argmax(values))
                reward = float(self.actions[action] * returns[index + 1])
                future = max(self.q_table.setdefault(int(states[index + 1]), [0.0, 0.0, 0.0]))
                values[action] += alpha * (reward + gamma * future - values[action])
        self.config["return_bins"] = bins.tolist()
        return {"states": float(len(self.q_table)), "episodes": float(episodes)}

    def generate(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if "return_bins" not in self.config:
            self.fit(rows)
        close = pd.to_numeric(pd.DataFrame(rows)["close"], errors="raise")
        states = np.digitize(close.pct_change().fillna(0).to_numpy(float), self.config["return_bins"])
        result = []
        for index, state in enumerate(states):
            values = self.q_table.get(int(state), [0.0, 0.0, 0.0])
            signal = self.actions[int(np.argmax(values))]
            result.append({"index": index, "signal": signal, "score": float(max(values)), "target_weight": float(signal)})
        return result

    def save(self, path: str | Path) -> dict[str, str]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"format_version": 1, "capability_id": self.capability_id, "config": self.config, "q_table": self.q_table}, sort_keys=True), encoding="utf-8")
        return {"artifact_uri": str(target), "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str = "") -> "QLearningStrategyRuntime":
        source = Path(path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("Strategy artifact SHA256 mismatch")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("capability_id") != cls.capability_id:
            raise ValueError("Strategy artifact capability mismatch")
        runtime = cls(payload["config"])
        runtime.q_table = {int(key): [float(value) for value in values] for key, values in payload["q_table"].items()}
        return runtime

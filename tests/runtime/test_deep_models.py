from __future__ import annotations

import pytest

from qbot_runtime import create_model, load_model


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("capability", ["qbot.mlp", "qbot.lstm", "qbot.gru", "qbot.transformer", "qbot.tft"])
def test_deep_model_round_trip(capability: str, tmp_path) -> None:
    path = tmp_path / f"{capability}.qbot"
    runtime = create_model(capability, {"epochs": 2, "hidden_size": 4, "heads": 2})
    training = runtime.fit([[0., 1.], [1., 2.], [2., 3.], [3., 4.]], [0., 1., 2., 3.], path)
    loaded = load_model(path, expected_sha256=training.artifact.metadata["artifact_sha256"])
    prediction = loaded.predict_batch([[4., 5.]])
    assert len(prediction.values) == 1
    assert prediction.metadata["capability_id"] == capability

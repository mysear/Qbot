from __future__ import annotations

import pytest

from qbot_runtime import create_model, load_model


def test_regression_model_round_trip(tmp_path) -> None:
    artifact_path = tmp_path / "linear.qbot"
    model = create_model("qbot.linear_regression")
    training = model.fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0], artifact_path)

    loaded = load_model(artifact_path, expected_sha256=training.artifact.metadata["artifact_sha256"])
    prediction = loaded.predict_batch([[3.0], [4.0]])

    assert prediction.values == pytest.approx([7.0, 9.0])
    assert prediction.probabilities is None
    assert prediction.metadata["provider"] == "qbot"


def test_classification_model_returns_positive_probabilities(tmp_path) -> None:
    model = create_model("qbot.logistic_regression")
    model.fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1], tmp_path / "classifier.qbot")

    prediction = model.predict_batch([[0.5], [2.5]])

    assert prediction.probabilities is not None
    assert all(0.0 <= value <= 1.0 for value in prediction.probabilities)


def test_model_rejects_feature_shape_change(tmp_path) -> None:
    model = create_model("qbot.linear_regression")
    model.fit([[0.0], [1.0]], [0.0, 1.0], tmp_path / "linear.qbot")

    with pytest.raises(ValueError, match="expected 1 features"):
        model.predict_batch([[1.0, 2.0]])


def test_unknown_capability_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown Qbot model capability"):
        create_model("qbot.not_real")


def test_seeded_model_is_reproducible(tmp_path) -> None:
    values = []
    for index in range(2):
        model = create_model("qbot.random_forest", {"random_state": 17, "n_estimators": 8})
        model.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0], tmp_path / f"seeded-{index}.qbot")
        values.append(model.predict_batch([[1.5], [2.5]]).values)
    assert values[0] == values[1]


def test_format_version_one_artifact_remains_loadable(tmp_path) -> None:
    path = tmp_path / "compatible.qbot"
    trained = create_model("qbot.linear_regression").fit([[0.0], [1.0]], [0.0, 1.0], path)
    assert load_model(path, expected_sha256=trained.artifact.metadata["artifact_sha256"]).predict_batch([[2.0]]).values == pytest.approx([2.0])

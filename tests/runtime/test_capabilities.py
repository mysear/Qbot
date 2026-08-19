from qbot_runtime import capabilities


def test_manifest_only_advertises_stable_runtime_capabilities() -> None:
    manifest = capabilities()

    assert manifest.runtime_version == "0.3.1"
    assert len(manifest.qbot_revision) == 40
    assert {item.id for item in manifest.capabilities} >= {
        "qbot.linear_regression",
        "qbot.logistic_regression",
        "qbot.lightgbm",
        "qbot.xgboost",
    }
    models = [item for item in manifest.capabilities if item.kind == "model"]
    assert all(item.supports_fit and item.supports_batch_predict for item in models)
    assert {item.kind for item in manifest.capabilities} >= {"model", "feature", "strategy", "backtest", "data", "execution"}
    assert all(item.display_name and item.config_schema is not None for item in models)
    assert not {"qbot.alpha101", "qbot.alpha191"} & {item.id for item in manifest.capabilities}


def test_optional_dependency_state_is_exposed() -> None:
    manifest = capabilities()
    catboost = next(item for item in manifest.capabilities if item.id == "qbot.catboost")

    assert catboost.available or catboost.unavailable_reason

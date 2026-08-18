# Qbot Runtime Package

`mysear-qbot-runtime` is the stable Python integration facade for applications
embedding Qbot. It intentionally excludes the desktop GUI, notebooks and trade
terminal dependencies from the core installation.

Install the core package from a pinned release or commit:

```bash
python -m pip install "mysear-qbot-runtime @ git+https://github.com/mysear/Qbot.git@<tag-or-commit>"
```

Optional groups provide tree models, deep-learning, backtest and data clients:

```bash
python -m pip install "mysear-qbot-runtime[tree] @ git+https://github.com/mysear/Qbot.git@<tag-or-commit>"
```

Applications should import only `qbot_runtime`. Imports from Qbot's internal
GUI, strategy and benchmark directories are not part of the stable contract.
Model artifacts use Python serialization and must only be loaded from trusted,
checksum-verified storage controlled by the application.

```python
from qbot_runtime import capabilities, create_model

manifest = capabilities()
model = create_model("qbot.lightgbm", {"task": "regression"})
result = model.fit(features, targets, "model.qbot")
predictions = model.predict_batch(future_features)
```

Only capabilities with implemented `fit`, `load` and `predict_batch` contracts
are advertised. A model mentioned in the main Qbot README is not automatically
available through this runtime facade.

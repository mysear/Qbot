"""Stable integration facade for Qbot capabilities."""

from .capabilities import Capability, CapabilityManifest, capabilities
from .models import ModelRuntime, create_model, load_model
from .features import FeatureRuntime, create_feature
from .strategies import StrategyRuntime, create_strategy
from .backtest import BacktestRuntime, create_backtest_engine
from .data import DataProvider, create_data_provider
from .execution import ExecutionClient, create_execution_client
from .schemas import BacktestResult, FeatureResult, ModelArtifact, OrderResult, PredictionResult, StrategyResult, TrainingResult

__all__ = [
    "Capability", "CapabilityManifest", "ModelArtifact", "ModelRuntime",
    "PredictionResult", "TrainingResult", "FeatureResult", "StrategyResult", "BacktestResult", "OrderResult",
    "FeatureRuntime", "StrategyRuntime", "BacktestRuntime", "DataProvider", "ExecutionClient",
    "capabilities", "create_model", "load_model", "create_feature", "create_strategy",
    "create_backtest_engine", "create_data_provider", "create_execution_client",
]

__version__ = "0.2.0"

"""Stable integration facade for Qbot capabilities."""

from .capabilities import Capability, CapabilityManifest, capabilities
from .models import ModelRuntime, create_model, load_model
from .schemas import ModelArtifact, PredictionResult, TrainingResult

__all__ = [
    "Capability", "CapabilityManifest", "ModelArtifact", "ModelRuntime",
    "PredictionResult", "TrainingResult", "capabilities", "create_model", "load_model",
]

__version__ = "0.1.0"

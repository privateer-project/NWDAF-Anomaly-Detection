# HITL Package Root
#
# This file makes the hitl directory a Python package and exposes
# the main public API for library usage.
#
# Should expose:
# - HITL orchestrator class (main entry point)
# - Key type definitions (PredictResult, TrainParams, SchemaInfo)
# - Custom exceptions for error handling
# - Version information
#
# Example usage after import:
#   from hitl import HITL, PredictResult, NoLiveModel
#   hitl = HITL(config)
#   result = hitl.filter_predict(tensor=data)
#
# Responsibilities:
# - Define __version__ = "0.1.0"
# - Import and re-export main classes from core.hitl
# - Import and re-export main types from types.py
# - Import and re-export main errors from errors.py
# - Define __all__ list for clean public API

__version__ = "0.1.0"

# Re-export main public API
from .core.hitl import HITL  # noqa: E402,F401
from .types import PredictResult, TrainParams, SchemaInfo  # noqa: E402,F401
from .errors import NoLiveModel, ShapeMismatch, UnsupportedShape  # noqa: E402,F401

__all__ = [
	"HITL",
	"PredictResult",
	"TrainParams",
	"SchemaInfo",
	"NoLiveModel",
	"ShapeMismatch",
	"UnsupportedShape",
]

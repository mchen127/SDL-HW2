"""Evaluation module for model testing and results collection."""

from .evaluator import Evaluator
from .results_handler import ResultsHandler

__all__ = [
    "MetricsComputer",
    "compute_top_k_error",
    "Evaluator",
    "ResultsHandler",
]

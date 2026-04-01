"""Baseline models for comparison experiments."""

from .lstm import LSTMReportGenerator
from .simple_cnn_lstm import SimpleCNNLSTM
from .transformer_baseline import TransformerBaseline

BASELINES = {
    "lstm": LSTMReportGenerator,
    "transformer": TransformerBaseline,
    "simple_cnn_lstm": SimpleCNNLSTM,
}

__all__ = [
    "BASELINES",
    "LSTMReportGenerator",
    "TransformerBaseline",
    "SimpleCNNLSTM",
]

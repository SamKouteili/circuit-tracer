"""
GNNExplainer package for Circuit-Tracer GraphGPS model

This package provides explainability tools for understanding
prompt injection detection decisions made by the GraphGPS model.
"""

from .gnn_explainer_core import CircuitTracerGNNExplainer
from .explanation import AttributionGraphExplanation, ExplanationBatch
from .circuit_tracer_processor import CircuitTracerExplanationProcessor
from .metrics import ExplanationEvaluator
from .visualization import ExplanationVisualizer
from .utils import ExplanationCache, BatchProcessor

__all__ = [
    'CircuitTracerGNNExplainer',
    'AttributionGraphExplanation',
    'ExplanationBatch',
    'CircuitTracerExplanationProcessor',
    'ExplanationEvaluator',
    'ExplanationVisualizer',
    'ExplanationCache',
    'BatchProcessor'
]

__version__ = '0.1.0'
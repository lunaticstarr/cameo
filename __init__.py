"""
CAMEO - Computational Annotation of Model Entities and Ontologies

A standalone LLM-powered tool for automated annotation of SBML and SBML-qual models.
"""

__version__ = "0.1.0"
__author__ = "Luna Li"
__email__ = "lixy2401@gmail.com"

# Import main interface functions
from .core import annotate_model, annotate_single_model, curate_model, curate_single_model, print_results

__all__ = [
    'annotate_model',         # For models without existing annotations
    'annotate_single_model', 
    'curate_model',           # For models with existing annotations
    'curate_single_model',
    'print_results'
] 
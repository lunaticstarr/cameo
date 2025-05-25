"""
CAMEO - Computational Annotation of Model Entities and Ontologies

A standalone LLM-powered tool for automated annotation of SBML and SBML-qual models.
"""

__version__ = "0.1.0"
__author__ = "Luna Li"
__email__ = "lixy2401@gmail.com"

# Import main interface functions
from .core import annotate_single_model, annotate_model, print_results

__all__ = [
    'annotate_single_model',
    'annotate_model', 
    'print_results'
] 
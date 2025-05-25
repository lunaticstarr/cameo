"""
CAMEO Core Module

Contains the main functionality for LLM-powered model annotation.
Streamlined interface following AMAS workflow.
"""

# Main annotation interface
from .annotation_workflow import annotate_model, annotate_single_model, print_results

# Individual components for advanced users
from .model_info import find_species_with_chebi_annotations, extract_model_info, format_prompt
from .llm_interface import SYSTEM_PROMPT, query_llm, parse_llm_response
from .database_search import get_species_recommendations_direct, search_database, get_available_databases, Recommendation

__all__ = [
    # Main interface - what most users will use
    'annotate_model',
    'annotate_single_model', 
    'print_results',
    
    # Individual components for advanced usage
    'find_species_with_chebi_annotations',
    'extract_model_info',
    'format_prompt',
    'SYSTEM_PROMPT',
    'query_llm',
    'parse_llm_response',
    'get_species_recommendations_direct',
    'search_database',
    'get_available_databases',
    'Recommendation'
] 
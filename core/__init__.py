"""
CAMEO Core Module
"""

# Main annotation interface - for models with no or limited annotations
from .annotation_workflow import annotate_model, annotate_single_model

# Main curation interface - for models with existing annotations
from .curation_workflow import curate_model, curate_single_model

# Shared functionality
from .annotation_workflow import print_results

# Individual components for advanced users
from .model_info import find_species_with_chebi_annotations, extract_model_info, format_prompt, get_species_display_names, get_all_species_ids
from .llm_interface import SYSTEM_PROMPT, query_llm, parse_llm_response
from .database_search import get_species_recommendations_direct, search_database, get_available_databases, Recommendation

__all__ = [
    # Main interfaces - what most users will use
    'annotate_model',  
    'annotate_single_model', 
    'curate_model', 
    'curate_single_model',
    'print_results',
    
    # Individual components
    'get_all_species_ids',
    'find_species_with_chebi_annotations',
    'get_species_display_names',
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
"""
CAMEO Utilities Module

Contains utility functions and constants used throughout CAMEO.
"""

from .constants import (
    EntityType, 
    ModelType, 
    DatabaseID,
    DATABASE_PREFIXES,
    DATABASE_URIS,
    ENTITY_DATABASE_MAPPING,
    DEFAULT_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD
)

from .evaluation import (
    evaluate_single_model,
    evaluate_models_in_folder,
    print_evaluation_results,
    calculate_species_statistics,
    compare_with_amas_results
)

__all__ = [
    # Constants
    'EntityType',
    'ModelType', 
    'DatabaseID',
    'DATABASE_PREFIXES',
    'DATABASE_URIS',
    'ENTITY_DATABASE_MAPPING',
    'DEFAULT_CONFIDENCE_THRESHOLD',
    'HIGH_CONFIDENCE_THRESHOLD',
    'LOW_CONFIDENCE_THRESHOLD',
    
    # Evaluation functions (for internal testing)
    'evaluate_single_model',
    'evaluate_models_in_folder',
    'print_evaluation_results',
    'calculate_species_statistics',
    'compare_with_amas_results'
] 
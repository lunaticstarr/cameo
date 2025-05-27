"""
Annotation Workflow for CAMEO

Main interface for annotating a single model that has no or limited existing annotations.
Provides the primary function that users will call to get recommendation tables
for all species in a model.
"""

import time
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import numpy as np

from .model_info import find_species_with_chebi_annotations, extract_model_info, format_prompt, get_species_display_names, get_all_species_ids
from .llm_interface import SYSTEM_PROMPT, query_llm, parse_llm_response
from .database_search import get_species_recommendations_direct, Recommendation

logger = logging.getLogger(__name__)



def annotate_single_model(model_file: str, 
                  llm_model: str = "gpt-4o-mini",
                  max_entities: int = 100,
                  entity_type: str = "chemical",
                  database: str = "chebi") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Annotate a single model that has no or limited existing annotations.
    
    This is the main function users will call to get annotation recommendations
    for all species in a model, regardless of existing annotations.
    
    Args:
        model_file: Path to SBML model file
        llm_model: LLM model to use ("gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct:free")
        max_entities: Maximum number of entities to annotate
        entity_type: Type of entities to annotate ("chemical", "gene", "protein")
        database: Target database ("chebi", "ncbigene", "uniprot")
        
    Returns:
        Tuple of (recommendations_df, metrics_dict)
        - recommendations_df: AMAS-compatible DataFrame with annotation recommendations
        - metrics_dict: Dictionary with evaluation metrics and timing information
    """
    start_time = time.time()
    
    logger.info(f"Starting annotation for model: {model_file}")
    logger.info(f"Using LLM model: {llm_model}")
    logger.info(f"Entity type: {entity_type}, Database: {database}")
    
    # Step 1: Get all species from model
    logger.info("Step 1: Getting all species from model...")
    all_species_ids = get_all_species_ids(model_file)
    
    if not all_species_ids:
        logger.warning("No species found in model")
        return pd.DataFrame(), {"error": "No species found in model"}
    
    logger.info(f"Found {len(all_species_ids)} species in model")
    
    # Step 2: Check for existing annotations (for metrics calculation)
    existing_annotations = {}
    if entity_type == "chemical" and database == "chebi":
        existing_annotations = find_species_with_chebi_annotations(model_file)
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    else:
        # Future: support other entity types and databases
        logger.warning(f"Entity type {entity_type} with database {database} not yet supported")
    
    # Step 3: Select entities to evaluate (limit if needed)
    specs_to_evaluate = all_species_ids[:max_entities]
    logger.info(f"Selected {len(specs_to_evaluate)} entities for annotation")
    
    # Step 4: Extract model context
    logger.info("Step 4: Extracting model context...")
    model_info = extract_model_info(model_file, specs_to_evaluate)
    
    if not model_info:
        logger.error("Failed to extract model context")
        return pd.DataFrame(), {"error": "Failed to extract model context"}
    
    logger.info(f"Extracted context for model: {model_info['model_name']}")
    
    # Step 5: Format prompt for LLM
    logger.info("Step 5: Formatting LLM prompt...")
    prompt = format_prompt(model_file, specs_to_evaluate)
    
    if not prompt:
        logger.error("Failed to format prompt")
        return pd.DataFrame(), {"error": "Failed to format prompt"}
    
    # Step 6: Query LLM
    logger.info(f"Step 6: Querying LLM ({llm_model})...")
    llm_start = time.time()
    
    try:
        result = query_llm(prompt, SYSTEM_PROMPT, model=llm_model)
        llm_time = time.time() - llm_start
        
        if not result:
            logger.error("No response from LLM")
            return pd.DataFrame(), {"error": "No response from LLM"}
        
        logger.info(f"LLM response received in {llm_time:.2f}s")
        
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return pd.DataFrame(), {"error": f"LLM query failed: {e}"}
    
    # Step 7: Parse LLM response
    logger.info("Step 7: Parsing LLM response...")
    synonyms_dict, reason = parse_llm_response(result)
    
    if not synonyms_dict:
        logger.error("Failed to parse LLM response")
        return pd.DataFrame(), {"error": "Failed to parse LLM response"}
    
    logger.info(f"Parsed synonyms for {len(synonyms_dict)} entities")
    
    # Step 8: Search database
    logger.info(f"Step 8: Searching {database} database...")
    search_start = time.time()
    
    if database == "chebi":
        recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict)
    else:
        # Future: support other databases
        logger.error(f"Database {database} not yet supported")
        return pd.DataFrame(), {"error": f"Database {database} not yet supported"}
    
    search_time = time.time() - search_start
    logger.info(f"Database search completed in {search_time:.2f}s")
    
    # Step 9: Generate recommendation table
    logger.info("Step 9: Generating recommendation table...")
    recommendations_df = _generate_recommendation_table(
        model_file, recommendations, existing_annotations, model_info
    )
    
    # Step 10: Calculate metrics
    total_time = time.time() - start_time
    metrics = _calculate_metrics(
        recommendations_df, existing_annotations, len(all_species_ids), total_time, llm_time, search_time
    )
    
    logger.info(f"Annotation completed in {total_time:.2f}s")
    logger.info(f"Generated {len(recommendations_df)} recommendations")
    
    return recommendations_df, metrics

def _generate_recommendation_table(model_file: str, 
                                 recommendations: List[Recommendation],
                                 existing_annotations: Dict[str, List[str]],
                                 model_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate AMAS-compatible recommendation table.
    
    Args:
        model_file: Path to model file
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations (may be empty)
        model_info: Model information dictionary
        
    Returns:
        DataFrame in AMAS format
    """
    rows = []
    filename = Path(model_file).name
    
    for rec in recommendations:
        if not rec.candidates:
            # No candidates found
            row = {
                'file': filename,
                'type': 'chemical',  # Currently only supporting chemicals
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'annotation': '',
                'annotation_label': '',
                'match_score': 0.0,
                'existing': 0,
                'update_annotation': 'ignore'
            }
            rows.append(row)
            continue
        
        # Add row for each candidate
        for i, candidate in enumerate(rec.candidates):
            # Determine if this is an existing annotation
            existing = 1 if candidate in existing_annotations.get(rec.id, []) else 0
            
            # Calculate match score (hits / total synonyms)
            match_score = rec.hit_count[i] / len(rec.synonyms) if rec.synonyms else 0
            
            # Determine update action - for new annotations, suggest adding top candidates
            if existing:
                update_action = 'keep'
            elif i == 0 and match_score > 0.5:  # Top candidate with good score
                update_action = 'add'
            else:
                update_action = 'ignore'
            
            row = {
                'file': filename,
                'type': 'chemical',
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'annotation': candidate,
                'annotation_label': rec.candidate_names[i],
                'match_score': match_score,
                'existing': existing,
                'update_annotation': update_action
            }
            rows.append(row)
    
    return pd.DataFrame(rows)

def _calculate_metrics(recommendations_df: pd.DataFrame,
                      existing_annotations: Dict[str, List[str]],
                      total_species: int,
                      total_time: float,
                      llm_time: float,
                      search_time: float) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for annotation workflow.
    
    Args:
        recommendations_df: Recommendation DataFrame
        existing_annotations: Dictionary of existing annotations (may be empty)
        total_species: Total number of species in the model
        total_time: Total processing time
        llm_time: LLM query time
        search_time: Database search time
        
    Returns:
        Dictionary with metrics
    """
    if recommendations_df.empty:
        return {
            'total_entities': total_species,
            'entities_with_predictions': 0,
            'annotation_rate': 0.0,
            'total_predictions': 0,
            'matches': 0,
            'accuracy': np.nan if not existing_annotations else 0.0,
            'total_time': total_time,
            'llm_time': llm_time,
            'search_time': search_time
        }
    
    entities_with_predictions = recommendations_df[recommendations_df['annotation'] != '']['id'].nunique()
    annotation_rate = entities_with_predictions / total_species if total_species > 0 else 0
    
    # Calculate accuracy based on existing annotations
    total_predictions = len(recommendations_df[recommendations_df['annotation'] != ''])
    matches = len(recommendations_df[recommendations_df['existing'] == 1])
    
    # If no existing annotations, accuracy is NA
    if not existing_annotations:
        accuracy = np.nan
    else:
        entities_with_existing = len(existing_annotations)
        accuracy = matches / entities_with_existing if entities_with_existing > 0 else 0
    
    return {
        'total_entities': total_species,
        'entities_with_predictions': entities_with_predictions,
        'annotation_rate': annotation_rate,
        'total_predictions': total_predictions,
        'matches': matches,
        'accuracy': accuracy,
        'total_time': total_time,
        'llm_time': llm_time,
        'search_time': search_time
    }

def print_results(results_df: pd.DataFrame):
    """
    Print evaluation results summary.
    Adapted from AMAS test_LLM_synonyms_plain.ipynb for annotation workflow
    
    Args:
        results_df: DataFrame with evaluation results
    """
    if results_df.empty:
        print("No results to display")
        return
    
    print("Number of models assessed: %d" % results_df['model'].nunique())
    print("Number of models with predictions: %d" % results_df[results_df['annotation'] != '']['model'].nunique())
    
    # Calculate per-model averages - handle NaN accuracy values
    model_accuracies = results_df.groupby('model')['existing'].mean()
    valid_accuracies = model_accuracies[~pd.isna(model_accuracies)]
    
    if len(valid_accuracies) > 0:
        print("Average accuracy (per model, where existing annotations available): %.02f" % valid_accuracies.mean())
    else:
        print("Average accuracy: N/A (no existing annotations)")
    
    mean_processing_time = results_df.groupby('model')['total_time'].first().mean()
    print("Ave. total time (per model): %.02f" % mean_processing_time)
    
    num_elements = results_df.groupby('model').size().mean()
    mean_processing_time_per_element = mean_processing_time / num_elements
    print("Ave. total time (per element, per model): %.02f" % mean_processing_time_per_element)
    
    # LLM time
    mean_llm_time = results_df.groupby('model')['llm_time'].first().mean()
    print("Ave. LLM time (per model): %.02f" % mean_llm_time)
    
    mean_llm_time_per_element = mean_llm_time / num_elements
    print("Ave. LLM time (per element, per model): %.02f" % mean_llm_time_per_element)
    
    # Average number of predictions per species
    average_predictions = results_df[results_df['annotation'] != ''].groupby('model').size().mean()
    print(f"Average number of predictions per model: {average_predictions}")

# Main interface function for users
def annotate_model(model_file: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main interface function for annotating a single model.
    
    This is the primary function users should call for models without existing annotations.
    
    Args:
        model_file: Path to SBML model file
        **kwargs: Additional arguments passed to annotate_single_model
        
    Returns:
        Tuple of (recommendations_df, metrics_dict)
    """
    return annotate_single_model(model_file, **kwargs) 
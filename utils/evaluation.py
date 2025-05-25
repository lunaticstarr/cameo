"""
Evaluation Utilities for CAMEO

Internal evaluation functions for testing and validation.
Replicates the evaluation workflow from AMAS test_LLM_synonyms_plain.ipynb.
"""

import os
import time
import pandas as pd
import numpy as np
import lzma
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from ..core.annotation_workflow import annotate_model
from ..core.model_info import find_species_with_chebi_annotations
from ..core.database_search import Recommendation

logger = logging.getLogger(__name__)

# Cache for loaded dictionaries
_CHEBI_LABEL_DICT: Optional[Dict[str, str]] = None
_CHEBI_FORMULA_DICT: Optional[Dict[str, str]] = None

def load_chebi_label_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to label dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to their labels
    """
    global _CHEBI_LABEL_DICT
    
    if _CHEBI_LABEL_DICT is None:
        data_file = Path(__file__).parent.parent / "data" / "chebi" / "chebi2label.lzma"
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_LABEL_DICT = pickle.load(f)
    
    return _CHEBI_LABEL_DICT

def load_chebi_formula_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to formula dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to their formulas
    """
    global _CHEBI_FORMULA_DICT
    
    if _CHEBI_FORMULA_DICT is None:
        data_file = Path(__file__).parent.parent / "data" / "chebi" / "chebi_shortened_formula_comp.lzma"
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI formula data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_FORMULA_DICT = pickle.load(f)
    
    return _CHEBI_FORMULA_DICT

def get_recall(ref: Dict[str, List[str]], pred: Dict[str, List[str]], mean: bool = True) -> float:
    """
    Calculate recall metric.
    Replicates tools.getRecall from AMAS.
    
    Args:
        ref: Reference annotations {id: [annotations]}
        pred: Predicted annotations {id: [annotations]}
        mean: If True, return average across all IDs
        
    Returns:
        Recall value(s)
    """
    ref_keys = set(ref.keys())
    pred_keys = set(pred.keys())
    species_to_test = ref_keys.intersection(pred_keys)
    recall = {}
    
    for one_k in species_to_test:
        num_intersection = len(set(ref[one_k]).intersection(pred[one_k]))
        recall[one_k] = num_intersection / len(set(ref[one_k])) if ref[one_k] else 0
    
    if mean:
        return np.round(np.mean([recall[val] for val in recall.keys()]), 3) if recall else 0.0
    else:
        return {val: np.round(recall[val], 3) for val in recall.keys()}

def get_precision(ref: Dict[str, List[str]], pred: Dict[str, List[str]], mean: bool = True) -> float:
    """
    Calculate precision metric.
    Replicates tools.getPrecision from AMAS.
    
    Args:
        ref: Reference annotations {id: [annotations]}
        pred: Predicted annotations {id: [annotations]}
        mean: If True, return average across all IDs
        
    Returns:
        Precision value(s)
    """
    ref_keys = set(ref.keys())
    pred_keys = set(pred.keys())
    precision = {}
    species_to_test = ref_keys.intersection(pred_keys)
    
    for one_k in species_to_test:
        num_intersection = len(set(ref[one_k]).intersection(pred[one_k]))
        num_predicted = len(set(pred[one_k]))
        if num_predicted == 0:
            precision[one_k] = 0.0
        else:
            precision[one_k] = num_intersection / num_predicted
    
    if mean:
        if precision:
            return np.round(np.mean([precision[val] for val in precision.keys()]), 3)
        else:
            return 0.0
    else:
        return {val: np.round(precision[val], 3) for val in precision.keys()}

def get_species_statistics(recommendations: List[Recommendation], 
                          refs_formula: Dict[str, List[str]], 
                          refs_chebi: Dict[str, List[str]], 
                          model_mean: bool = False) -> Dict[str, Any]:
    """
    Calculate species statistics including formula and ChEBI-based metrics.
    Replicates getSpeciesStatistics from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        recommendations: List of Recommendation objects
        refs_formula: Reference formulas {id: [formulas]}
        refs_chebi: Reference ChEBI IDs {id: [chebi_ids]}
        model_mean: If True, return model-level averages
        
    Returns:
        Dictionary with recall and precision statistics
    """
    # Convert recommendations to prediction dictionaries
    preds_chebi = {val.id: [k for k in val.candidates] for val in recommendations}
    
    # Convert ChEBI predictions to formulas
    formula_dict = load_chebi_formula_dict()
    preds_formula = {}
    for k in preds_chebi.keys():
        formulas = []
        for chebi_id in preds_chebi[k]:
            if chebi_id in formula_dict:
                formula = formula_dict[chebi_id]
                if formula:  # Only add non-empty formulas
                    formulas.append(formula)
        preds_formula[k] = formulas
    
    # Calculate metrics
    recall_formula = get_recall(ref=refs_formula, pred=preds_formula, mean=model_mean)
    precision_formula = get_precision(ref=refs_formula, pred=preds_formula, mean=model_mean)
    recall_chebi = get_recall(ref=refs_chebi, pred=preds_chebi, mean=model_mean)
    precision_chebi = get_precision(ref=refs_chebi, pred=preds_chebi, mean=model_mean)
    
    return {
        'recall_formula': recall_formula, 
        'recall_chebi': recall_chebi, 
        'precision_formula': precision_formula, 
        'precision_chebi': precision_chebi
    }

def evaluate_single_model(model_file: str, 
                         llm_model: str = 'meta-llama/llama-3.3-70b-instruct:free',
                         max_entities: int = None,
                         entity_type: str = "chemical",
                         database: str = "chebi",
                         save_llm_results: bool = True,
                         output_dir: str = './results/') -> Optional[pd.DataFrame]:
    """
    Generate species evaluation statistics for one model.
    
    Args:
        model_file: Path to SBML model file
        llm_model: LLM model to use
        max_entities: Maximum number of entities to evaluate (None for all)
        entity_type: Type of entities to annotate
        database: Target database
        save_llm_results: Whether to save LLM results to files
        output_dir: Directory to save results
        
    Returns:
        DataFrame with evaluation results or None if failed
    """
    try:
        model_name = Path(model_file).name
        logger.info(f"Evaluating model: {model_name}")
        
        # Get existing annotations to determine entities to evaluate
        if entity_type == "chemical" and database == "chebi":
            existing_annotations = find_species_with_chebi_annotations(model_file)
        else:
            logger.warning(f"Entity type {entity_type} with database {database} not yet supported")
            return None
        
        if not existing_annotations:
            logger.warning(f"No existing annotations found in {model_name}")
            return None
        
        # Limit entities if specified
        specs_to_evaluate = list(existing_annotations.keys())
        if max_entities:
            specs_to_evaluate = specs_to_evaluate[:max_entities]
        
        logger.info(f"Evaluating {len(specs_to_evaluate)} entities in {model_name}")
        
        # Run annotation with access to LLM results
        from ..core.model_info import extract_model_info, format_prompt
        from ..core.llm_interface import SYSTEM_PROMPT, query_llm, parse_llm_response
        from ..core.database_search import get_species_recommendations_direct
        
        # Extract model context and query LLM
        model_info = extract_model_info(model_file, specs_to_evaluate)
        prompt = format_prompt(model_file, specs_to_evaluate)
        
        # Query LLM and get response
        llm_start = time.time()
        llm_response = query_llm(prompt, SYSTEM_PROMPT, model=llm_model)
        llm_time = time.time() - llm_start
        
        # Parse LLM response
        synonyms_dict, reason = parse_llm_response(llm_response)
        
        # Search database
        search_start = time.time()
        recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict)
        search_time = time.time() - search_start
        
        total_time = llm_time + search_time
        
        if not recommendations:
            logger.warning(f"No recommendations generated for {model_name}")
            return None
        
        # Convert to AMAS-compatible format with LLM results
        result_df = _convert_format(
            recommendations, existing_annotations, model_name, 
            synonyms_dict, reason, total_time, llm_time, search_time
        )
        
        # Save LLM results if requested
        if save_llm_results:
            _save_llm_results(model_file, llm_model, output_dir, synonyms_dict, reason)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Failed to evaluate model {model_file}: {e}")
        return None

def evaluate_models_in_folder(model_dir: str,
                             num_models: str = 'all',
                             llm_model: str = 'meta-llama/llama-3.3-70b-instruct:free',
                             max_entities: int = None,
                             entity_type: str = "chemical",
                             database: str = "chebi",
                             save_llm_results: bool = True,
                             output_dir: str = './results/',
                             output_file: str = 'evaluation_results.csv',
                             start_at: int = 1) -> pd.DataFrame:
    """
    Generate species evaluation statistics for multiple models in a directory.
    Replicates evaluate_models from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        model_dir: Directory containing SBML model files
        num_models: Number of models to evaluate ('all' or integer)
        llm_model: LLM model to use
        max_entities: Maximum entities per model (None for all)
        entity_type: Type of entities to annotate
        database: Target database
        save_llm_results: Whether to save LLM results
        output_dir: Directory to save results
        output_file: Name of output CSV file
        start_at: Model index to start at (1-based)
        
    Returns:
        Combined DataFrame with all evaluation results
    """
    # Get model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.xml')]
    model_files.sort()  # Ensure consistent ordering
    
    # Determine which models to evaluate
    if num_models == 'all':
        num_models = len(model_files)
        model_files = model_files[start_at-1:]
    else:
        num_models = int(min(num_models, len(model_files) - start_at + 1))
        model_files = model_files[start_at-1:start_at+num_models-1]
    
    logger.info(f"Evaluating {len(model_files)} models starting from index {start_at}")
    
    # Initialize result storage
    all_results = []
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    for idx, model_file in enumerate(model_files):
        actual_idx = idx + start_at
        logger.info(f"Evaluating {actual_idx}/{start_at + len(model_files) - 1}: {model_file}")
        
        model_path = os.path.join(model_dir, model_file)
        
        # Evaluate single model
        result_df = evaluate_single_model(
            model_file=model_path,
            llm_model=llm_model,
            max_entities=max_entities,
            entity_type=entity_type,
            database=database,
            save_llm_results=save_llm_results,
            output_dir=output_dir
        )
        
        if result_df is not None:
            all_results.append(result_df)
            
            # Save intermediate results
            intermediate_file = output_path / f"{output_file}_{actual_idx}.csv"
            result_df.to_csv(intermediate_file, index=False)
            logger.info(f"Saved intermediate results to: {intermediate_file}")
        else:
            logger.warning(f"Skipping {model_file} - no results generated")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save final results
        final_file = output_path / output_file
        combined_df.to_csv(final_file, index=False)
        logger.info(f"Saved final results to: {final_file}")
        
        return combined_df
    else:
        logger.warning("No results generated for any models")
        return pd.DataFrame()

def _convert_format(recommendations: List[Recommendation],
                                   existing_annotations: Dict[str, List[str]],
                                   model_name: str,
                                   synonyms_dict: Dict[str, List[str]],
                                   reason: str,
                                   total_time: float,
                                   llm_time: float,
                                   search_time: float) -> pd.DataFrame:
    """
    Convert CAMEO recommendations to AMAS evaluation format with LLM results.
    
    Args:
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations
        model_name: Name of the model file
        synonyms_dict: LLM-generated synonyms
        reason: LLM reasoning
        total_time: Total processing time
        llm_time: LLM query time
        search_time: Database search time
        
    Returns:
        DataFrame in AMAS evaluation format
    """
    # Load required dictionaries
    label_dict = load_chebi_label_dict()
    formula_dict = load_chebi_formula_dict()
    
    # Prepare reference data for statistics calculation
    refs_chebi = existing_annotations
    refs_formula = {}
    for species_id, chebi_ids in existing_annotations.items():
        formulas = []
        for chebi_id in chebi_ids:
            if chebi_id in formula_dict:
                formula = formula_dict[chebi_id]
                if formula:
                    formulas.append(formula)
        refs_formula[species_id] = formulas
    
    # Calculate statistics
    stats = get_species_statistics(recommendations, refs_formula, refs_chebi, model_mean=False)
    
    # Convert to AMAS format
    result_rows = []
    for rec in recommendations:
        species_id = rec.id
        
        # Get existing annotation names
        existing_ids = existing_annotations.get(species_id, [])
        existing_names = [label_dict.get(chebi_id, chebi_id) for chebi_id in existing_ids]
        exist_annotation_name = ', '.join(existing_names) if existing_names else 'NA'
        
        # Get LLM synonyms
        llm_synonyms = synonyms_dict.get(species_id, [])
        
        # Get predictions and their names
        predictions = rec.candidates
        prediction_names = [label_dict.get(chebi_id, chebi_id) for chebi_id in predictions]
        
        # Calculate match scores (hits / total synonyms)
        match_scores = []
        if rec.hit_count and llm_synonyms:
            match_scores = [hit / len(llm_synonyms) for hit in rec.hit_count]
        else:
            match_scores = [0.0] * len(predictions)
                
        # Get statistics for this species
        recall_formula = stats['recall_formula'].get(species_id, 0) if isinstance(stats['recall_formula'], dict) else 0
        precision_formula = stats['precision_formula'].get(species_id, 0) if isinstance(stats['precision_formula'], dict) else 0
        recall_chebi = stats['recall_chebi'].get(species_id, 0) if isinstance(stats['recall_chebi'], dict) else 0
        precision_chebi = stats['precision_chebi'].get(species_id, 0) if isinstance(stats['precision_chebi'], dict) else 0

        # Calculate accuracy (1 if recall_formula > 0, 0 otherwise)
        accuracy = 1 if recall_formula > 0 else 0
        
        # Create row in AMAS format
        row = {
            'model': model_name,
            'species_id': species_id,
            'display_name': rec.synonyms[0] if rec.synonyms else species_id,  # Use first synonym as display name
            'synonyms_LLM': llm_synonyms,
            'reason': reason,
            'exist_annotation_chebi': existing_ids,
            'exist_annotation_name': exist_annotation_name,
            'predictions': predictions,
            'predictions_names': prediction_names,
            'match_score': match_scores,
            'recall_formula': recall_formula,
            'precision_formula': precision_formula,
            'recall_chebi': recall_chebi,
            'precision_chebi': precision_chebi,
            'accuracy': accuracy,
            'total_time': total_time,
            'llm_time': llm_time,
            'query_time': search_time
        }
        result_rows.append(row)
    
    return pd.DataFrame(result_rows)

def _save_llm_results(model_file: str, llm_model: str, output_dir: str, 
                     synonyms_dict: Dict[str, List[str]], reason: str):
    """
    Save LLM results to file.
    
    Args:
        model_file: Path to model file
        llm_model: LLM model used
        output_dir: Output directory
        synonyms_dict: LLM-generated synonyms
        reason: LLM reasoning
    """
    model_name = Path(model_file).stem
    if llm_model == "meta-llama/llama-3.3-70b-instruct:free":
        llm_name = "llama-3.3-70b-instruct"
    else:
        llm_name = llm_model

    output_dir = output_dir+llm_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / f"{model_name}_llm_results.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"LLM: {llm_model}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Synonyms:\n")
        for species_id, synonyms in synonyms_dict.items():
            f.write(f"{species_id}: {synonyms}\n")
        f.write(f"\nReason: {reason}\n")
    print(f"LLM results saved to: {output_file}")

def print_evaluation_results(results_csv: str):
    """
    Print evaluation results summary.
    Replicates print_results from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        results_csv: Path to results CSV file
    """
    if not os.path.exists(results_csv):
        print(f"Results file not found: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    if df.empty:
        print("No results to display")
        return
    
    print("Number of models assessed: %d" % df['model'].nunique())
    print("Number of models with predictions: %d" % df[df['predictions'] != '[]']['model'].nunique())
    
    # Calculate per-model averages
    model_accuracy = df.groupby('model')['accuracy'].mean().mean()
    print("Average accuracy (per model): %.02f" % model_accuracy)
    
    mean_processing_time = df.groupby('model')['total_time'].first().mean()
    print("Ave. total time (per model): %.02f" % mean_processing_time)
    
    num_elements = df.groupby('model').size().mean()
    mean_processing_time_per_element = mean_processing_time / num_elements if num_elements > 0 else 0
    print("Ave. total time (per element, per model): %.02f" % mean_processing_time_per_element)
    
    # LLM time
    mean_llm_time = df.groupby('model')['llm_time'].first().mean()
    print("Ave. LLM time (per model): %.02f" % mean_llm_time)
    
    mean_llm_time_per_element = mean_llm_time / num_elements if num_elements > 0 else 0
    print("Ave. LLM time (per element, per model): %.02f" % mean_llm_time_per_element)
    
    # Average number of predictions per species
    def safe_eval_predictions(x):
        """Safely evaluate predictions string."""
        try:
            if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
                return eval(x)
            elif isinstance(x, list):
                return x
            else:
                return []
        except Exception:
            return []
    
    df['parsed_predictions'] = df['predictions'].apply(safe_eval_predictions)
    df['num_predictions'] = df['parsed_predictions'].apply(len)
    average_predictions = df['num_predictions'].mean()
    print(f"Average number of predictions per species: {average_predictions:.2f}")

def calculate_species_statistics(recommendations: List[Recommendation],
                                existing_annotations: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate evaluation statistics for species recommendations.
    Simplified version of getSpeciesStatistics from AMAS.
    
    Args:
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations
        
    Returns:
        Dictionary with recall and precision statistics
    """
    stats = {}
    
    for rec in recommendations:
        species_id = rec.id
        predicted_ids = rec.candidates
        existing_ids = existing_annotations.get(species_id, [])
        
        # Calculate simple recall and precision
        if existing_ids:
            # Recall: fraction of existing annotations that were predicted
            matches = set(predicted_ids) & set(existing_ids)
            recall = len(matches) / len(existing_ids) if existing_ids else 0
        else:
            recall = 0
        
        if predicted_ids:
            # Precision: fraction of predictions that match existing annotations
            matches = set(predicted_ids) & set(existing_ids)
            precision = len(matches) / len(predicted_ids) if predicted_ids else 0
        else:
            precision = 0
        
        stats[species_id] = {
            'recall_chebi': recall,
            'precision_chebi': precision,
            'recall_formula': 0,  # Not implemented
            'precision_formula': 0  # Not implemented
        }
    
    return stats

def compare_with_amas_results(cameo_results_csv: str, amas_results_csv: str) -> pd.DataFrame:
    """
    Compare CAMEO results with AMAS results.
    
    Args:
        cameo_results_csv: Path to CAMEO results CSV
        amas_results_csv: Path to AMAS results CSV
        
    Returns:
        DataFrame with comparison statistics
    """
    if not os.path.exists(cameo_results_csv):
        raise FileNotFoundError(f"CAMEO results file not found: {cameo_results_csv}")
    
    if not os.path.exists(amas_results_csv):
        raise FileNotFoundError(f"AMAS results file not found: {amas_results_csv}")
    
    cameo_df = pd.read_csv(cameo_results_csv)
    amas_df = pd.read_csv(amas_results_csv)
    
    # Calculate summary statistics
    cameo_stats = {
        'models_assessed': cameo_df['model'].nunique(),
        'avg_accuracy': cameo_df.groupby('model')['accuracy'].mean().mean(),
        'avg_total_time': cameo_df.groupby('model')['total_time'].first().mean(),
        'avg_llm_time': cameo_df.groupby('model')['llm_time'].first().mean()
    }
    
    amas_stats = {
        'models_assessed': amas_df['model'].nunique(),
        'avg_accuracy': amas_df.groupby('model')['accuracy'].mean().mean() if 'accuracy' in amas_df.columns else 0,
        'avg_total_time': amas_df.groupby('model')['total_time'].first().mean() if 'total_time' in amas_df.columns else 0,
        'avg_llm_time': amas_df.groupby('model')['llm_time'].first().mean() if 'llm_time' in amas_df.columns else 0
    }
    
    comparison = pd.DataFrame({
        'Metric': ['Models Assessed', 'Average Accuracy', 'Average Total Time', 'Average LLM Time'],
        'CAMEO': [cameo_stats['models_assessed'], cameo_stats['avg_accuracy'], 
                 cameo_stats['avg_total_time'], cameo_stats['avg_llm_time']],
        'AMAS': [amas_stats['models_assessed'], amas_stats['avg_accuracy'],
                amas_stats['avg_total_time'], amas_stats['avg_llm_time']]
    })
    
    return comparison 
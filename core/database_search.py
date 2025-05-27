"""
Database Search for CAMEO

Handles database searches for annotation candidates.
Currently supports ChEBI, extensible to other databases.
"""

import os
import re
import lzma
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Cache for loaded dictionaries
_CHEBI_CLEANNAMES_DICT: Optional[Dict[str, List[str]]] = None
_CHEBI_LABEL_DICT: Optional[Dict[str, str]] = None

@dataclass
class Recommendation:
    """
    Recommendation dataclass
    """
    id: str  # ID for the species
    synonyms: list  # List of synonyms predicted by LLM
    candidates: list  # List of ChEBI IDs
    candidate_names: list  # List of names of the predicted candidates
    hit_count: list  # Number of hits in the synonyms

def get_data_dir() -> Path:
    """Get the path to the CAMEO data directory."""
    current_dir = Path(__file__).parent.parent
    return current_dir / "data" / "chebi"

def load_chebi_cleannames_dict() -> Dict[str, List[str]]:
    """
    Load the ChEBI clean names to ChEBI ID dictionary.
    
    Returns:
        Dictionary mapping clean names to lists of ChEBI IDs
    """
    global _CHEBI_CLEANNAMES_DICT
    
    if _CHEBI_CLEANNAMES_DICT is None:
        data_file = get_data_dir() / "cleannames2chebi.lzma"
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI cleannames data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_CLEANNAMES_DICT = pickle.load(f)
    
    return _CHEBI_CLEANNAMES_DICT

def load_chebi_label_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to label dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to their labels
    """
    global _CHEBI_LABEL_DICT
    
    if _CHEBI_LABEL_DICT is None:
        data_file = get_data_dir() / "chebi2label.lzma"
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_LABEL_DICT = pickle.load(f)
    
    return _CHEBI_LABEL_DICT

def remove_symbols(text: str) -> str:
    """
    Remove all characters except numbers and letters.
    
    Args:
        text: Input text to clean
        
    Returns:
        Text with only alphanumeric characters
    """
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def get_species_recommendations_direct(species_ids: List[str], synonyms_dict) -> List[Recommendation]:
    """
    Find ChEBI recommendations by directly matching against ChEBI synonyms.
    
    Parameters:
    - species_ids (list): List of species IDs to evaluate.
    - synonyms_dict (dict): Mapping of species IDs to synonyms.
    
    Returns:
    - list: List of Recommendation objects with candidates and names.
    """
    cleannames_dict = load_chebi_cleannames_dict()
    label_dict = load_chebi_label_dict()
    
    recommendations = []
    
    for spec_id in species_ids:
        # Get synonyms for this species ID
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(spec_id, [spec_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            # If it's a tuple with two items (dict and reason)
            synonyms = synonyms_dict[0].get(spec_id, [spec_id])
        else:
            synonyms = [spec_id]
        
        all_candidates = []
        all_candidate_names = []
        hit_count = {}  # Dictionary to track how many times each candidate appears
        
        # Query for each synonym
        for synonym in synonyms:
            norm_synonym = remove_symbols(synonym.lower())
            # Check all entries in cleannames dict for matches
            for ref_name, chebi_ids in cleannames_dict.items():
                if norm_synonym == ref_name.lower():
                    for chebi_id in chebi_ids:
                        chebi_name = label_dict.get(chebi_id, chebi_id)
                        
                        if chebi_id not in all_candidates:
                            all_candidates.append(chebi_id)
                            all_candidate_names.append(chebi_name)
                            hit_count[chebi_id] = 1
                        else:
                            hit_count[chebi_id] += 1
        
        # Convert hit_count dict to list in the same order as candidates
        hit_count_list = [hit_count.get(candidate, 0) for candidate in all_candidates]
        
        # Create recommendation object
        recommendation = Recommendation(
            id=spec_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            hit_count=hit_count_list
        )
        recommendations.append(recommendation)
    
    return recommendations

def search_database(entity_name: str, 
                   entity_type: str, 
                   database: str = "chebi",
                   max_candidates: int = 10) -> List[Tuple[str, float, str]]:
    """
    Search for annotation candidates in specified database.
    Currently supports ChEBI, extensible to other databases.
    
    Args:
        entity_name: Name of entity to search for
        entity_type: Type of entity (chemical, gene, protein)
        database: Database to search in (currently only "chebi")
        max_candidates: Maximum number of candidates to return
        
    Returns:
        List of tuples (database_id, confidence, description)
    """
    if database.lower() == "chebi":
        return _search_chebi(entity_name, max_candidates)
    else:
        logger.warning(f"Database {database} not yet supported")
        return []

def _search_chebi(entity_name: str, max_candidates: int = 10) -> List[Tuple[str, float, str]]:
    """
    Search ChEBI database for entity matches.
    
    Args:
        entity_name: Name to search for
        max_candidates: Maximum number of candidates
        
    Returns:
        List of tuples (chebi_id, confidence, description)
    """
    try:
        cleannames_dict = load_chebi_cleannames_dict()
        label_dict = load_chebi_label_dict()
        
        # Normalize entity name
        norm_name = remove_symbols(entity_name.lower())
        
        candidates = []
        
        # Direct match search
        for ref_name, chebi_ids in cleannames_dict.items():
            if norm_name == ref_name.lower():
                for chebi_id in chebi_ids:
                    chebi_name = label_dict.get(chebi_id, chebi_id)
                    confidence = 1.0  # Direct match gets highest confidence
                    candidates.append((chebi_id, confidence, chebi_name))
        
        # Partial match search if no direct matches
        if not candidates:
            for ref_name, chebi_ids in cleannames_dict.items():
                if norm_name in ref_name.lower() or ref_name.lower() in norm_name:
                    for chebi_id in chebi_ids:
                        chebi_name = label_dict.get(chebi_id, chebi_id)
                        # Calculate confidence based on string similarity
                        confidence = min(len(norm_name), len(ref_name.lower())) / max(len(norm_name), len(ref_name.lower()))
                        candidates.append((chebi_id, confidence, chebi_name))
        
        # Sort by confidence and limit results
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
        
    except Exception as e:
        logger.error(f"ChEBI search failed for {entity_name}: {e}")
        return []

def is_database_available(database: str) -> bool:
    """
    Check if a database is available for searching.
    
    Args:
        database: Database name to check
        
    Returns:
        True if database is available
    """
    if database.lower() == "chebi":
        try:
            data_dir = get_data_dir()
            cleannames_file = data_dir / "cleannames2chebi.lzma"
            labels_file = data_dir / "chebi2label.lzma"
            return cleannames_file.exists() and labels_file.exists()
        except Exception:
            return False
    
    return False

def get_available_databases() -> List[str]:
    """
    Get list of available databases.
    
    Returns:
        List of available database names
    """
    available = []
    
    if is_database_available("chebi"):
        available.append("chebi")
    
    # Future databases can be added here
    # if is_database_available("ncbigene"):
    #     available.append("ncbigene")
    
    return available 
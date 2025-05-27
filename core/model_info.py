"""
Model Information Extraction for CAMEO

Extracts model information and context for annotation, following AMAS workflow.
Consolidates functionality from model_parser.py and model_context.py.
"""

import re
import libsbml
import antimony
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def find_species_with_chebi_annotations(model_file: str) -> Dict[str, List[str]]:
    """
    Find species with existing ChEBI annotations.
    
    Args:
        model_file: Path to the SBML model file
        
    Returns:
        Dictionary mapping species IDs to their ChEBI annotation IDs
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return {}
    
    chebi_annotations = {}
    
    # Extract annotations from species
    for species in model.getListOfSpecies():
        species_id = species.getId()
        
        if species.isSetAnnotation():
            annotation = species.getAnnotation()
            # Parse RDF annotation to extract identifiers.org URIs
            annotation_str = annotation.toXMLString()
            
            # Look for identifiers.org ChEBI URIs (both http and https, and both forms)
            chebi_pattern = r'http[s]?://identifiers\.org/chebi/CHEBI:(\d+)'
            chebi_matches = re.findall(chebi_pattern, annotation_str)

            if chebi_matches:
                chebi_ids = [f"CHEBI:{match}" for match in chebi_matches]
                chebi_annotations[species_id] = chebi_ids
    
    return chebi_annotations

def get_species_display_names(model_file: str) -> Dict[str, str]:
    """
    Get the display names for all species in the model using libsbml.
    
    Args:
        model_file: Path to the SBML model file
        
    Returns:
        Dictionary mapping species IDs to their display names
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return {}
    
    names = {val.getId(): val.getName() for val in model.getListOfSpecies()}
    return names

def get_all_species_ids(model_file: str) -> List[str]:
    """
    Get all species IDs from an SBML model.
    
    Args:
        model_file: Path to SBML model file
        
    Returns:
        List of species IDs
    """
    display_names = get_species_display_names(model_file)
    return list(display_names.keys())
    
def extract_model_info(model_file: str, species_ids: List[str]) -> Dict[str, Any]:
    """
    Extract display names and reactions for the specified species from antimony.
    Also includes display names of all species involved in reactions with the specified species.
    
    Args:
        model_file: Path to the SBML model file
        species_ids: List of species IDs to include
        
    Returns:
        Dictionary with model name, display names, and reactions or empty dict if file cannot be loaded
    """
    # Load SBML file using antimony
    antimony.clearPreviousLoads()
    sbml_model = antimony.loadSBMLFile(model_file)
    if sbml_model == -1:
        print(f"Error loading SBML file: {antimony.getLastError()}")
        return {}  # Return empty dictionary instead of None
    
    antimony_string = antimony.getAntimonyString()
    
    # Extract model name
    model_name = ""
    # Look for model name in format "model *ModelName()"
    model_pattern = re.search(r'model \*(.*?)\(\)', antimony_string)
    if model_pattern:
        model_name = model_pattern.group(1)
    
    # Extract model notes that appear after "model notes" and are surrounded by ```
    model_notes_pattern = re.search(r'model notes\s*```(.*?)```', antimony_string, re.DOTALL)
    if model_notes_pattern:
        model_notes = model_notes_pattern.group(1)
        # Remove HTML tags from model notes
        model_notes = re.sub(r'<[^>]*>', '', model_notes)

        # Split by newlines and/or multiple spaces
        lines = re.split(r'\n|\s{2,}', model_notes)
        
        # List of keywords/fragments that indicate boilerplate text
        boilerplate_keywords = [
            'copyright', 'public domain', 'rights', 'CC0', 'dedication', 
            'please refer', 'BioModels Database', 'cite', 'citing',
            'terms of use', 'Li C', 'BMC Syst', 'encoded model', 
            'entitled to use', 'redistribute', 'commercially', 'restricted way', 'verbatim',
            'BIOMD', 'resource'
        ]
        
        # Filter out lines containing boilerplate keywords
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if line and not any(keyword.lower() in line.lower() for keyword in boilerplate_keywords):
                filtered_lines.append(line)
        
        # Reassemble the filtered content with proper spacing
        model_notes = '\n'.join(filtered_lines)
        model_notes = f"```{model_notes}```"
    else:
        model_notes = ""

    # Get display names from SBML for all species
    all_display_names = get_species_display_names(model_file)
    
    # Extract reactions involving our species
    reactions = []
    
    # Parse the antimony_string to extract reactions
    # Look for lines with => symbols which indicate reactions
    reaction_pattern = re.compile(r'// Reactions:.*?(?=//|$)', re.DOTALL)
    reactions_section = reaction_pattern.search(antimony_string)
    
    reaction_matches = []
    if reactions_section:
        reactions_text = reactions_section.group(0).replace("// Reactions:", "").strip()
        # Now extract individual reactions with => symbols
        reaction_pattern = re.compile(r'([^;]+)(=>)([^;]+);', re.MULTILINE)
        reaction_matches = reaction_pattern.findall(reactions_text)

    # If no matches found with '=>', try with '=' instead
    if not reaction_matches:
        reaction_pattern = re.compile(r'// Rate Rules:.*?(?=//|$)', re.DOTALL)
        reactions_section = reaction_pattern.search(antimony_string)
        
        reaction_matches = []
        if reactions_section:
            reactions_text = reactions_section.group(0).replace("// Rate Rules:", "").strip()
            # Now extract individual reactions with => symbols
            reaction_pattern = re.compile(r'([^;]+)(=)([^;]+);', re.MULTILINE)
            reaction_matches = reaction_pattern.findall(reactions_text)

    # Keep track of all species involved in reactions with our target species
    related_species = set(species_ids)
    
    # Filter reactions to only include those involving our species
    for match in reaction_matches:
        left_side, arrow, right_side = match
        reaction_str = f"{left_side.strip()} {arrow} {right_side.strip()}"
        
        # Check if any of our species IDs are in this reaction
        if any(re.search(r'\b' + re.escape(species_id) + r'\b', left_side + ' ' + right_side) for species_id in species_ids):
            reactions.append(reaction_str)
            
            # Extract all species IDs from this reaction
            all_ids_in_reaction = re.findall(r'\b([A-Za-z0-9_]+)\b', left_side + ' ' + right_side)
            related_species.update(all_ids_in_reaction)
    
    # Filter display names to include our target species and all related species
    filtered_display_names = {species_id: all_display_names.get(species_id, "") for species_id in related_species if species_id in all_display_names}
    
    return {
        "model_name": model_name,
        "display_names": filtered_display_names,
        "reactions": reactions,
        "model_notes": model_notes
    }

def format_prompt(model_file: str, species_ids: List[str]) -> str:
    """
    Format the information for the LLM prompt.
    
    Args:
        model_file: Path to the SBML model file
        species_ids: List of species IDs to include in the prompt
        
    Returns:
        Formatted prompt string
    """
    model_info = extract_model_info(model_file, species_ids)
    if model_info == {}:
        return ""
    
    prompt = f"""Now annotate these:
Species to annotate: {", ".join(species_ids)}
Model: "{model_info["model_name"]}"
// Display Names:
{model_info["display_names"]}
// Reactions:
{model_info["reactions"]}
// Notes:
{model_info["model_notes"]}

Return up to 3 standardized names or common synonyms for each species, ranked by likelihood.
Use the below format, do not include any other text except the synonyms, and give short reasons for all species after 'Reason:'

SpeciesA: "name1", "name2", …
SpeciesB:  …
Reason: …
    """
    return prompt 
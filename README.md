# CAMEO (Computational Annotation of Model Entities and Ontologies)

CAMEO is a LLM-powered system for annotating biosimulation models with standardized ontology terms.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Environment Variables

Set up your LLM provider API keys:

```bash
# For OpenAI models (gpt-4o-mini, gpt-4.1-nano)
export OPENAI_API_KEY="your-openai-key"

# For OpenRouter models (meta-llama/llama-3.3-70b-instruct:free)
export OPENROUTER_API_KEY="your-openrouter-key"
```

## Usage

CAMEO currently provides two main workflows:

### 1. Annotation Workflow (for new models)

For models with no or limited existing annotations. Annotates all species in the model:

```python
from cameo.core import annotate_model

# Annotate all species in a model
recommendations_df, metrics = annotate_model(
    model_file="path/to/model.xml",
    llm_model="gpt-4o-mini"
)

print(f"Total entities: {metrics['total_entities']}")
print(f"Annotation rate: {metrics['annotation_rate']:.1%}")
if not pd.isna(metrics['accuracy']):
    print(f"Accuracy: {metrics['accuracy']:.1%}")
else:
    print("Accuracy: N/A (no existing annotations)")
print(f"Total time: {metrics['total_time']:.2f}s")

# Save results
recommendations_df.to_csv("annotation_results.csv", index=False)
```

### 2. Curation Workflow (for models with existing annotations)

For models that already have annotations. Evaluates and improves existing annotations:

```python
from cameo.core import curate_model

# Curate existing annotations
curations_df, metrics = curate_model(
    model_file="path/to/model.xml",
    llm_model="gpt-4o-mini"
)

print(f"Entities with existing annotations: {metrics['total_entities']}")
print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Total time: {metrics['total_time']:.2f}s")

# Save results
curations_df.to_csv("curation_results.csv", index=False)
```

### Advanced Usage

```python
# More control over parameters
recommendations_df, metrics = annotate_model(
    model_file="model.xml",
    llm_model="meta-llama/llama-3.3-70b-instruct:free",
    max_entities=50,
    entity_type="chemical",
    database="chebi"
)
```

### Example

```python
# Using "tests/test_models/BIOMD0000000190.xml"
python examples/simple_example.py
```

## Workflows

### Annotation Workflow

- **Purpose**: Annotate models with no or limited existing annotations
- **Input**: All species in the model
- **Output**: Annotation recommendations for all species
- **Metrics**: Accuracy is NA when no existing annotations available

### Curation Workflow

- **Purpose**: Evaluate and improve existing annotations
- **Input**: Only species that already have annotations
- **Output**: Validation and improvement recommendations
- **Metrics**: Accuracy calculated against existing annotations

## Databases

### Currently Supported

- **ChEBI**: Chemical Entities of Biological Interest
  - Direct dictionary matching using compressed files
  - Synonym normalization and hit counting

### Future Support

- **NCBI Gene**: Gene annotation
- **UniProt**: Protein annotation
- **Rhea**: Reaction annotation
- **GO**: Gene Ontology terms

## Configuration

For more, see [config.yaml](config.yaml)

### LLM Model Selection

```python
# OpenAI models
llm_model = "gpt-4o-mini"      # Fast and cost-effective
llm_model = "gpt-4.1-nano"   

# OpenRouter models  
llm_model = "meta-llama/llama-3.3-70b-instruct:free"  # Free Llama model
```

### Entity Types and Databases

```python
# Currently supported
entity_type = "chemical"
database = "chebi"

# Future support
entity_type = "gene"      # database = "ncbigene"
entity_type = "protein"   # database = "uniprot"
```

## Data Files

### ChEBI Data

- **Location**: `data/chebi/`
- **Files**:
  - `cleannames2chebi.lzma`: Mapping from clean names to ChEBI IDs
  - `chebi2label.lzma`: Mapping from ChEBI IDs to labels
- **Source**: ChEBI ontology downloaded from https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl.gz.

## File Structure

```
cameo/
├── core/
│   ├── __init__.py              # Main interface exports
│   ├── annotation_workflow.py   # Annotation workflow (models without annotations)
│   ├── curation_workflow.py     # Curation workflow (models with annotations)
│   ├── model_info.py           # Model parsing and context
│   ├── llm_interface.py        # LLM interaction
│   └── database_search.py      # Database search functions
├── utils/
│   ├── constants.py
│   ├── evaluation.py 		# functions for evaluation
├── examples/
│   ├── simple_example.py    # Simple usage demo
├── data/
│   └── chebi/                   # ChEBI compressed dictionaries
└── tests/
    └── test_models     	 # Test models
    └── cameo_evaluation.ipynb   # evaluation notebook
```

## Future Development

### Planned Features

- **Multi-Database Support**: NCBI Gene, UniProt, GO, Rhea
- **RAG Integration**: Replace dictionary matching with vector embeddings
- **Web Interface**: User-friendly annotation tool

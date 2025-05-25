# CAMEO (Computational Annotation of Model Entities and Ontologies)

A standalone LLM-powered tool for SBML model annotation, following the AMAS workflow with a clean, streamlined interface.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Test installation
python examples/simple_annotation_example.py
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

### Usage

```python
from cameo.core import annotate_single_model

# Single function call for complete annotation
recommendations_df, metrics = annotate_single_model(
    model_file="path/to/model.xml",
    llm_model="gpt-4o-mini"
)

print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Annotation rate: {metrics['annotation_rate']:.1%}")
print(f"Total time: {metrics['total_time']:.2f}s")

# Save results
recommendations_df.to_csv("results.csv", index=False)
```

### Advanced Usage

```python
from cameo.core import annotate_model

# More control over parameters
recommendations_df, metrics = annotate_model(
    model_file="model.xml",
    llm_model="meta-llama/llama-3.3-70b-instruct:free",
    max_entities=50,
    entity_type="chemical",
    database="chebi"
)
```

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
│   ├── annotation_workflow.py   # Primary workflow functions
│   ├── model_info.py           # Model parsing and context
│   ├── llm_interface.py        # LLM interaction
│   └── database_search.py      # Database search functions
├── utils/
│   ├── constants.py
│   ├── evaluation.py 		# functions for evaluation
├── examples/
│   ├── simple_annotation_example.py    # Simple usage demo
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

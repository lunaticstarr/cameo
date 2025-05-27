#!/usr/bin/env python3
"""
Simple CAMEO Annotation Example

Demonstrates the streamlined core interface for annotating models.
Shows both annotation (for models without annotations) and curation (for models with existing annotations).
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main CAMEO interfaces
from cameo.core import annotate_model, curate_model, print_results

def main():
    """
    Simple example of using CAMEO to annotate and curate models.
    """
    print("CAMEO Simple Annotation & Curation Example")
    print("=" * 50)
    
    # Configuration
    model_file = "tests/test_models/BIOMD0000000190.xml"
    llm_model = "meta-llama/llama-3.3-70b-instruct:free"  # or "gpt-4o-mini"
    
    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        print("Please ensure the test model is available.")
        return
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        print("Warning: No API keys found in environment.")
        print("Set OPENAI_API_KEY or OPENROUTER_API_KEY to use LLM features.")
        return
    
    print(f"Model file: {model_file}")
    print(f"LLM model: {llm_model}")
    print()
    
    try:
        # Example 1: Curation workflow (for models with existing annotations)
        print("EXAMPLE 1: Curation workflow (models with existing annotations)")
        print("-" * 60)
        
        recommendations_df, metrics = curate_model(
            model_file=model_file,
            llm_model=llm_model,
            max_entities=5,  # Limit for demo
            entity_type="chemical",
            database="chebi"
        )
        
        # Display curation results
        if not recommendations_df.empty:
            print("Curation Results:")
            print(f"Total entities with existing annotations: {metrics['total_entities']}")
            print(f"Entities with predictions: {metrics['entities_with_predictions']}")
            print(f"Accuracy: {metrics['accuracy']:.1%}")
            print(f"Total time: {metrics['total_time']:.2f}s")
            print()
            
            # Show sample recommendations
            print("Sample Curation Recommendations:")
            sample_df = recommendations_df[['id', 'display_name', 'annotation', 'annotation_label', 'match_score', 'existing']].head(5)
            print(sample_df.to_string(index=False))
            print()
        else:
            if 'error' in metrics:
                print(f"Curation failed: {metrics['error']}")
        
        print("\n" + "="*60 + "\n")
        
        # Example 2: Annotation workflow (for models without existing annotations)
        print("EXAMPLE 2: Annotation workflow (all species, regardless of existing annotations)")
        print("-" * 80)
        
        recommendations_df2, metrics2 = annotate_model(
            model_file=model_file,
            llm_model=llm_model,
            max_entities=5,  # Limit for demo
            entity_type="chemical",
            database="chebi"
        )
        
        # Display annotation results
        if not recommendations_df2.empty:
            print("Annotation Results:")
            print(f"Total entities in model: {metrics2['total_entities']}")
            print(f"Entities with predictions: {metrics2['entities_with_predictions']}")
            print(f"Annotation rate: {metrics2['annotation_rate']:.1%}")
            
            if not pd.isna(metrics2['accuracy']):
                print(f"Accuracy (where existing annotations available): {metrics2['accuracy']:.1%}")
            else:
                print("Accuracy: N/A (no existing annotations to compare against)")
            
            print(f"Total time: {metrics2['total_time']:.2f}s")
            print()
            
            # Show sample recommendations
            print("Sample Annotation Recommendations:")
            sample_df2 = recommendations_df2[['id', 'display_name', 'annotation', 'annotation_label', 'match_score', 'existing']].head(5)
            print(sample_df2.to_string(index=False))
            print()
            
            # Save results
            output_file = "simple_annotation_results.csv"
            recommendations_df2.to_csv(output_file, index=False)
            print(f"Full annotation results saved to: {output_file}")
            
        else:
            print("No annotation recommendations generated.")
            if 'error' in metrics2:
                print(f"Error: {metrics2['error']}")
    
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
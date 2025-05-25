#!/usr/bin/env python3
"""
Simple CAMEO Annotation Example

Demonstrates the streamlined core interface for annotating a single model.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main CAMEO interface
from cameo.core import annotate_single_model, print_results

def main():
    """
    Simple example of using CAMEO to annotate a model.
    """
    print("CAMEO Simple Annotation Example")
    print("=" * 40)
    
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
        # This is the main function call - everything else is handled internally
        recommendations_df, metrics = annotate_single_model(
            model_file=model_file,
            llm_model=llm_model,
            max_entities=5,  # Limit for demo
            entity_type="chemical",
            database="chebi"
        )
        
        # Display results
        if not recommendations_df.empty:
            print("Annotation Results:")
            print("-" * 40)
            print(f"Total entities: {metrics['total_entities']}")
            print(f"Entities with predictions: {metrics['entities_with_predictions']}")
            print(f"Annotation rate: {metrics['annotation_rate']:.1%}")
            print(f"Accuracy: {metrics['accuracy']:.1%}")
            print(f"Total time: {metrics['total_time']:.2f}s")
            print(f"LLM time: {metrics['llm_time']:.2f}s")
            print(f"Search time: {metrics['search_time']:.2f}s")
            print()
            
            # Show sample recommendations
            print("Sample Recommendations:")
            print("-" * 40)
            sample_df = recommendations_df[['id', 'display_name', 'annotation', 'annotation_label', 'match_score', 'existing']].head(10)
            print(sample_df.to_string(index=False))
            print()
            
            # Save results
            output_file = "simple_annotation_results.csv"
            recommendations_df.to_csv(output_file, index=False)
            print(f"Full results saved to: {output_file}")
            
        else:
            print("No recommendations generated.")
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
    
    except Exception as e:
        print(f"Annotation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
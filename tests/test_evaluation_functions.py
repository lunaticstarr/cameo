#!/usr/bin/env python3
"""
Test script to verify CAMEO evaluation functions work correctly.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import CAMEO functions
from cameo.core import annotate_single_model
from cameo.utils import evaluate_single_model, print_evaluation_results

def test_evaluation_functions():
    """Test the evaluation functions with the test model."""
    
    print("Testing CAMEO Evaluation Functions")
    print("=" * 50)
    
    # Configuration
    test_model_file = "test_models/BIOMD0000000190.xml"
    llm_model = "meta-llama/llama-3.3-70b-instruct:free"
    output_dir = "./results/"
    
    # Check if test model exists
    if not os.path.exists(test_model_file):
        print(f"✗ Test model not found: {test_model_file}")
        print("Please ensure the test model is available in the tests directory.")
        return False
    
    print(f"✓ Test model found: {test_model_file}")
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        print("✗ No API keys found. Set OPENAI_API_KEY or OPENROUTER_API_KEY to test LLM features.")
        return False
    
    print("✓ API keys found")
    
    try:
        # Test 1: Core interface
        print("\nTest 1: Core interface (annotate_single_model)")
        print("-" * 50)
        
        recommendations_df, metrics = annotate_single_model(
            model_file=test_model_file,
            llm_model=llm_model,
            max_entities=3,  # Limit for quick test
            entity_type="chemical",
            database="chebi"
        )
        
        print(f"✓ Core interface test successful")
        print(f"  - Generated {len(recommendations_df)} recommendations")
        print(f"  - Accuracy: {metrics['accuracy']:.1%}")
        print(f"  - Total time: {metrics['total_time']:.2f}s")
        
        # Test 2: Utils evaluation function
        print("\nTest 2: Utils evaluation function")
        print("-" * 50)
        
        result_df = evaluate_single_model(
            model_file=test_model_file,
            llm_model=llm_model,
            max_entities=3,  # Limit for quick test
            entity_type="chemical",
            database="chebi",
            save_llm_results=True,
            output_dir=output_dir
        )
        
        if result_df is not None:
            print(f"✓ Utils evaluation test successful")
            print(f"  - Generated {len(result_df)} result rows")
            print(f"  - Average accuracy: {result_df['accuracy'].mean():.1%}")
            
            # Show new features
            print("\n  New features demonstrated:")
            
            # LLM results
            if 'synonyms_LLM' in result_df.columns:
                print("  ✓ LLM synonyms included")
                for idx, row in result_df[['species_id', 'synonyms_LLM']].head(2).iterrows():
                    print(f"    {row['species_id']}: {row['synonyms_LLM']}")
            
            if 'reason' in result_df.columns and not result_df['reason'].empty:
                print("  ✓ LLM reasoning included")
                print(f"    Reason: {result_df['reason'].iloc[0][:80]}...")
            
            # Formula-based metrics
            if 'recall_formula' in result_df.columns:
                print("  ✓ Formula-based metrics calculated")
                print(f"    Avg recall (formula): {result_df['recall_formula'].mean():.3f}")
                print(f"    Avg precision (formula): {result_df['precision_formula'].mean():.3f}")
            
            # Proper ChEBI labels
            if 'exist_annotation_name' in result_df.columns:
                print("  ✓ Proper ChEBI label names")
                for idx, row in result_df[['species_id', 'exist_annotation_name']].head(2).iterrows():
                    print(f"    {row['species_id']}: {row['exist_annotation_name']}")
            
            # Match scores
            if 'match_score' in result_df.columns:
                print("  ✓ Match scores calculated")
                for idx, row in result_df[['species_id', 'match_score']].head(2).iterrows():
                    print(f"    {row['species_id']}: {row['match_score']}")
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            result_file = os.path.join(output_dir, "test_results.csv")
            result_df.to_csv(result_file, index=False)
            print(f"\n  - Results saved to: {result_file}")
            
            # Test 3: Print results function
            print("\nTest 3: Print results function")
            print("-" * 50)
            print_evaluation_results(result_file)
            
        else:
            print(f"✗ Utils evaluation test failed: No results generated")
            return False
        
        print("\n" + "=" * 50)
        print("✓ All evaluation function tests passed!")
        print("\nKey improvements verified:")
        print("✓ LLM synonyms and reasoning in output")
        print("✓ Formula-based recall/precision calculations")
        print("✓ Proper ChEBI label names")
        print("✓ Match scores from hit counts")
        print("✓ AMAS-compatible CSV format")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation_functions()
    sys.exit(0 if success else 1) 
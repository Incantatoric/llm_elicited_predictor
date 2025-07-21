#!/usr/bin/env python3
"""
Test runner for Naive LLM evaluator
"""

from naive_llm_evaluator import NaiveLLMEvaluator
import sys
import os


def test_naive_llm():
    """Test Naive LLM evaluator with in-context learning"""
    print("Testing Naive LLM evaluator with in-context learning...")
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return None
    
    evaluator = NaiveLLMEvaluator()
    metrics = evaluator.run_evaluation()
    
    print(f"\nNaive LLM Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R-squared: {metrics['r_squared']:.4f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    if 'coverage_90' in metrics:
        print(f"90% Coverage: {metrics['coverage_90']:.2%}")
    
    return metrics


if __name__ == "__main__":
    try:
        metrics = test_naive_llm()
        
        if metrics:
            print("\n✅ Naive LLM test completed successfully!")
        else:
            print("\n❌ Test failed - check API key setup")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
#!/usr/bin/env python3
"""
Test runner for Bayesian evaluator
"""

from bayesian_evaluator import BayesianEvaluator
import sys
from pathlib import Path


def test_elicited_priors():
    """Test Bayesian evaluator with LLM-elicited priors"""
    print("Testing Bayesian evaluator with LLM-elicited priors...")
    
    evaluator = BayesianEvaluator(prior_type="elicited")
    metrics = evaluator.run_evaluation()
    
    print(f"\nElicited Priors Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R-squared: {metrics['r_squared']:.4f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    if 'coverage_95' in metrics:
        print(f"95% Coverage: {metrics['coverage_95']:.2%}")
    
    return metrics


def test_uninformed_priors():
    """Test Bayesian evaluator with uninformed priors"""
    print("\n" + "="*80)
    print("Testing Bayesian evaluator with uninformed priors...")
    
    evaluator = BayesianEvaluator(prior_type="uninformed")
    metrics = evaluator.run_evaluation()
    
    print(f"\nUninformed Priors Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R-squared: {metrics['r_squared']:.4f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    if 'coverage_95' in metrics:
        print(f"95% Coverage: {metrics['coverage_95']:.2%}")
    
    return metrics


def compare_results(elicited_metrics, uninformed_metrics):
    """Compare results between elicited and uninformed priors"""
    print("\n" + "="*80)
    print("COMPARISON: ELICITED vs UNINFORMED PRIORS")
    print("="*80)
    
    print(f"{'Metric':<20} {'Elicited':<12} {'Uninformed':<12} {'Difference':<12}")
    print("-" * 60)
    
    for metric in ['mae', 'rmse', 'r_squared', 'directional_accuracy']:
        if metric in elicited_metrics and metric in uninformed_metrics:
            elicited_val = elicited_metrics[metric]
            uninformed_val = uninformed_metrics[metric]
            diff = elicited_val - uninformed_val
            print(f"{metric:<20} {elicited_val:>10.4f} {uninformed_val:>10.4f} {diff:>10.4f}")
    
    if 'coverage_95' in elicited_metrics and 'coverage_95' in uninformed_metrics:
        elicited_val = elicited_metrics['coverage_95']
        uninformed_val = uninformed_metrics['coverage_95']
        diff = elicited_val - uninformed_val
        print(f"{'coverage_95':<20} {elicited_val:>10.4f} {uninformed_val:>10.4f} {diff:>10.4f}")


if __name__ == "__main__":
    try:
        # Test both methods
        elicited_metrics = test_elicited_priors()
        uninformed_metrics = test_uninformed_priors()
        
        # Compare results
        compare_results(elicited_metrics, uninformed_metrics)
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
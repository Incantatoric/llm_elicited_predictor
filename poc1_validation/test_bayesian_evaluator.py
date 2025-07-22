#!/usr/bin/env python3
"""
Test runner for Bayesian evaluator with different prior types
"""

from bayesian_evaluator import BayesianEvaluator
import sys
from pathlib import Path


def test_elicited_priors_expert():
    """Test Bayesian evaluator with expert LLM-elicited priors"""
    print("Testing Bayesian evaluator with expert LLM-elicited priors...")
    
    evaluator = BayesianEvaluator(prior_type="elicited", prior_folder="expert_10")
    metrics = evaluator.run_evaluation()
    
    print(f"\nExpert Priors Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R-squared: {metrics['r_squared']:.4f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    if 'coverage_95' in metrics:
        print(f"95% Coverage: {metrics['coverage_95']:.2%}")
    
    return metrics


def test_elicited_priors_data_informed_10():
    """Test Bayesian evaluator with data-informed LLM-elicited priors (10x10)"""
    print("\n" + "="*80)
    print("Testing Bayesian evaluator with data-informed LLM-elicited priors (10x10)...")
    
    evaluator = BayesianEvaluator(prior_type="elicited", prior_folder="data_informed_10")
    metrics = evaluator.run_evaluation()
    
    print(f"\nData-informed (10x10) Priors Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R-squared: {metrics['r_squared']:.4f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    if 'coverage_95' in metrics:
        print(f"95% Coverage: {metrics['coverage_95']:.2%}")
    
    return metrics


def test_elicited_priors_data_informed_3():
    """Test Bayesian evaluator with data-informed LLM-elicited priors (3x3)"""
    print("\n" + "="*80)
    print("Testing Bayesian evaluator with data-informed LLM-elicited priors (3x3)...")
    
    evaluator = BayesianEvaluator(prior_type="elicited", prior_folder="data_informed_3")
    metrics = evaluator.run_evaluation()
    
    print(f"\nData-informed (3x3) Priors Results:")
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


def compare_all_results(expert_metrics, data_informed_10_metrics, data_informed_3_metrics, uninformed_metrics):
    """Compare results between all prior types"""
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON: ALL PRIOR TYPES")
    print("="*80)
    
    print(f"{'Metric':<20} {'Expert (10x10)':<15} {'Data-informed (10x10)':<20} {'Data-informed (3x3)':<18} {'Uninformed':<12}")
    print("-" * 90)
    
    for metric in ['mae', 'rmse', 'r_squared', 'directional_accuracy']:
        expert_val = expert_metrics[metric]
        data_10_val = data_informed_10_metrics[metric]
        data_3_val = data_informed_3_metrics[metric]
        uninformed_val = uninformed_metrics[metric]
        
        print(f"{metric:<20} {expert_val:>13.4f} {data_10_val:>18.4f} {data_3_val:>16.4f} {uninformed_val:>10.4f}")
    
    if 'coverage_95' in expert_metrics:
        expert_val = expert_metrics['coverage_95']
        data_10_val = data_informed_10_metrics['coverage_95']
        data_3_val = data_informed_3_metrics['coverage_95']
        uninformed_val = uninformed_metrics['coverage_95']
        
        print(f"{'coverage_95':<20} {expert_val:>13.4f} {data_10_val:>18.4f} {data_3_val:>16.4f} {uninformed_val:>10.4f}")


if __name__ == "__main__":
    try:
        # Test all prior types
        expert_metrics = test_elicited_priors_expert()
        data_informed_10_metrics = test_elicited_priors_data_informed_10()
        data_informed_3_metrics = test_elicited_priors_data_informed_3()
        uninformed_metrics = test_uninformed_priors()
        
        # Compare all results
        compare_all_results(expert_metrics, data_informed_10_metrics, data_informed_3_metrics, uninformed_metrics)
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
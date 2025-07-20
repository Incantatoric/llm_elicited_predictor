#!/usr/bin/env python3
"""
Test the fixed Bayesian evaluator for uninformed priors
"""

from bayesian_evaluator import BayesianEvaluator
import pandas as pd


def test_uninformed_priors_only():
    """Test just the uninformed priors to see the fixed prior-posterior comparison"""
    print("Testing FIXED uninformed priors evaluator...")
    
    evaluator = BayesianEvaluator(prior_type="uninformed")
    metrics = evaluator.run_evaluation()
    
    # Check if the prior-posterior comparison file was created
    comparison_file = evaluator.results_dir / "prior_posterior_comparison.csv"
    if comparison_file.exists():
        print("\n📊 PRIOR-POSTERIOR COMPARISON FILE:")
        df = pd.read_csv(comparison_file)
        print(df.round(4))
    else:
        print("❌ Prior-posterior comparison file not found")
    
    return metrics


if __name__ == "__main__":
    test_uninformed_priors_only() 
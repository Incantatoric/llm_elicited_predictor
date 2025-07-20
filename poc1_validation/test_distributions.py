#!/usr/bin/env python3
"""
Test distribution visualizations for elicited priors
"""

from bayesian_evaluator import BayesianEvaluator


def test_elicited_distributions():
    """Test elicited priors with distribution visualizations"""
    print("Testing elicited priors with distribution visualizations...")
    
    evaluator = BayesianEvaluator(prior_type="elicited")
    
    # Run just the basic evaluation to get the model and trace
    evaluator.load_data()
    evaluator.create_train_test_split()
    evaluator.train_and_predict()
    evaluator.calculate_metrics()
    
    # Now test the distribution visualization
    evaluator.analyze_parameters()
    
    # Test individual visualization methods
    print("\n🎨 Testing distribution visualizations...")
    evaluator.create_distribution_plots()
    
    print("\n✅ Distribution visualization test complete!")


if __name__ == "__main__":
    test_elicited_distributions() 
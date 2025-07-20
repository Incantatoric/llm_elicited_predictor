#!/usr/bin/env python3
"""
Main execution script for 한화솔루션 LLM-elicited Bayesian stock prediction
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Import your modules
from src.data_collection import HanwhaDataCollector
from src.llm_elicitation import HanwhaLLMElicitor
from src.bayesian_model import HanwhaBayesianModel, create_uninformative_model
from src.interpretability import HanwhaInterpreter, BUSINESS_SCENARIOS

def main():
    """Main execution pipeline"""
    
    # 1. Data collection (if not already done)
    if not Path("data/features_standardized.csv").exists():
        print("Collecting data from Yahoo Finance...")
        collector = HanwhaDataCollector()
        X, y, feature_names, _ = collector.collect_and_prepare_data()
    else:
        print("Loading existing data...")
        X = pd.read_csv("data/features_standardized.csv", index_col=0)
        y = pd.read_csv("data/target_returns.csv", index_col=0).squeeze()
        with open("data/metadata.json") as f:
            metadata = json.load(f)
        feature_names = metadata['feature_names']
    
    # 2. LLM Prior elicitation (if not already done)
    if not Path("priors/hanwha_prior_0.npy").exists():
        print("Eliciting priors from LLM...")
        # You'll need to provide your API key
        api_key = input("Enter OpenAI API key: ")
        elicitor = HanwhaLLMElicitor(api_key=api_key)
        priors, prompt_info = elicitor.elicit_all_priors(feature_names)
        prior_arrays = elicitor.save_priors(priors, prompt_info, feature_names)
    else:
        print("Loading existing priors...")
    
    # 3. Train Bayesian model with elicited priors
    print("Training Bayesian model with LLM-elicited priors...")
    bayesian_model = HanwhaBayesianModel()
    
    # Split data for training (use last 6 months for testing)
    split_idx = len(X) - 6
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train model
    trace = bayesian_model.train_model(X_train.values, y_train.values)
    
    # 4. Make predictions
    print("Making predictions...")
    predictions = bayesian_model.predict(X_test.values)
    
    # 5. Interpretability analysis
    print("Generating interpretability analysis...")
    interpreter = HanwhaInterpreter(feature_names)
    
    # Load priors for visualization
    priors = bayesian_model.priors
    
    # Plot prior distributions
    interpreter.plot_prior_distributions(priors, "results/prior_distributions.png")
    
    # Generate parameter explanations
    explanations = interpreter.explain_parameter_effects(trace, priors)
    
    # Scenario analysis
    scenario_results = interpreter.scenario_analysis(
        bayesian_model.model, trace, feature_names, BUSINESS_SCENARIOS
    )
    
    # 6. Print business-friendly results
    print("\n" + "="*60)
    print("HANWHA SOLUTIONS STOCK PREDICTION RESULTS")
    print("="*60)
    
    print("\n📊 MODEL PREDICTIONS (Next 6 months):")
    print(f"Expected return: {predictions['mean'].mean():.2%}")
    print(f"Volatility: {predictions['std'].mean():.2%}")
    print(f"95% confidence interval: [{predictions['quantiles']['5%'].mean():.2%}, {predictions['quantiles']['95%'].mean():.2%}]")
    
    print("\n🧠 KEY FACTOR INTERPRETATIONS:")
    for feature, explanation in explanations.items():
        print(f"\n{feature}:")
        print(f"  {explanation['business_interpretation']}")
    
    print("\n🎯 SCENARIO ANALYSIS:")
    print(scenario_results.round(3))
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open("results/explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)
    
    scenario_results.to_csv("results/scenario_analysis.csv")
    
    print(f"\n📁 Results saved to {results_dir}/")
    print("Ready for boss presentation! 🎉")

if __name__ == "__main__":
    main()
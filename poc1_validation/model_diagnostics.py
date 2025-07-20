#!/usr/bin/env python3
"""
Comprehensive model diagnostics and results analysis
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import your modules using absolute imports
from hanwha_predictor.data.collector import HanwhaDataCollector
from hanwha_predictor.models.bayesian import HanwhaBayesianModel
from hanwha_predictor.analysis.interpretability import HanwhaInterpreter

def comprehensive_analysis():
    """Complete analysis of what we actually have"""
    
    print("="*80)
    print("COMPREHENSIVE MODEL DIAGNOSTICS AND RESULTS")
    print("="*80)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # 1. Load and examine the data
    print("\n1. DATA EXAMINATION")
    print("-" * 40)
    
    X = pd.read_csv(project_root / 'data/processed/features_standardized.csv', index_col=0)
    y = pd.read_csv(project_root / 'data/processed/target_returns.csv', index_col=0).squeeze()
    
    with open(project_root / 'data/processed/metadata.json') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']
    
    print(f"Total data points: {len(X)}")
    print(f"Features: {feature_names}")
    print(f"Date range: {X.index[0]} to {X.index[-1]}")
    print(f"Target statistics: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Show the actual split I used
    split_idx = len(X) - 6
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTRAIN/TEST SPLIT (what I implemented):")
    print(f"Training data: {len(X_train)} points ({X_train.index[0]} to {X_train.index[-1]})")
    print(f"Test data: {len(X_test)} points ({X_test.index[0]} to {X_test.index[-1]})")
    
    # 2. Model training and predictions
    print("\n2. MODEL TRAINING AND PREDICTIONS")
    print("-" * 40)
    
    model = HanwhaBayesianModel()
    print(f"Loaded {len(model.priors)} priors")
    
    # Train model
    trace = model.train_model(X_train.values, y_train.values, n_samples=1000, n_chains=2)
    
    # Get predictions for each test point individually
    predictions = model.predict(X_test.values)
    
    print(f"\nPredictions shape: {predictions['samples'].shape}")
    print(f"Mean predictions shape: {predictions['mean'].shape}")
    
    # 3. Individual month predictions
    print("\n3. INDIVIDUAL MONTH PREDICTIONS")
    print("-" * 40)
    
    results_df = pd.DataFrame({
        'Date': X_test.index,
        'Actual_Return': y_test.values,
        'Predicted_Mean': predictions['mean'],
        'Predicted_Std': predictions['std'],
        'Lower_5%': predictions['quantiles']['5%'],
        'Upper_95%': predictions['quantiles']['95%'],
        'Prediction_Error': y_test.values - predictions['mean']
    })
    
    print(results_df.round(4))
    
    # 4. Model fit statistics
    print("\n4. MODEL FIT STATISTICS")
    print("-" * 40)
    
    # Basic statistics
    mae = np.mean(np.abs(results_df['Prediction_Error']))
    mse = np.mean(results_df['Prediction_Error']**2)
    rmse = np.sqrt(mse)
    
    # R-squared equivalent
    ss_res = np.sum(results_df['Prediction_Error']**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    
    # Coverage probability (how often actual falls within prediction interval)
    in_interval = ((results_df['Actual_Return'] >= results_df['Lower_5%']) & 
                   (results_df['Actual_Return'] <= results_df['Upper_95%']))
    coverage = in_interval.mean()
    print(f"95% Prediction Interval Coverage: {coverage:.2%}")
    
    # 5. Bayesian model diagnostics
    print("\n5. BAYESIAN MODEL DIAGNOSTICS")
    print("-" * 40)
    
    try:
        import arviz as az
        
        # R-hat (convergence diagnostic)
        rhat = az.rhat(trace)
        print("R-hat values (should be < 1.1):")
        for var in rhat.data_vars:
            rhat_val = float(rhat[var].values) if rhat[var].values.size == 1 else rhat[var].values.mean()
            print(f"  {var}: {rhat_val:.3f}")
        
        # Effective sample size
        ess = az.ess(trace)
        print("\nEffective Sample Size:")
        for var in ess.data_vars:
            ess_val = float(ess[var].values) if ess[var].values.size == 1 else ess[var].values.mean()
            print(f"  {var}: {ess_val:.0f}")
            
    except ImportError:
        print("ArviZ not available for advanced diagnostics")
    
    # 6. Parameter estimates
    print("\n6. PARAMETER ESTIMATES")
    print("-" * 40)
    
    # Extract parameter estimates
    posterior_summary = {}
    for i, feature in enumerate(['intercept'] + feature_names):
        param_name = f'beta_{i}'
        if param_name in trace.posterior.data_vars:
            samples = trace.posterior[param_name].values.flatten()
            posterior_summary[feature] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'lower_95': np.percentile(samples, 2.5),
                'upper_95': np.percentile(samples, 97.5)
            }
    
    param_df = pd.DataFrame(posterior_summary).T
    print(param_df.round(4))
    
    # 7. Residual analysis
    print("\n7. RESIDUAL ANALYSIS")
    print("-" * 40)
    
    residuals = results_df['Prediction_Error']
    print(f"Residual mean: {residuals.mean():.4f} (should be ~0)")
    print(f"Residual std: {residuals.std():.4f}")
    
    # Test for normality (Shapiro-Wilk test)
    try:
        from scipy import stats
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"Shapiro-Wilk test p-value: {shapiro_p:.4f} (>0.05 suggests normal residuals)")
    except ImportError:
        print("SciPy not available for normality test")
    
    # 8. Prior vs Posterior comparison
    print("\n8. PRIOR VS POSTERIOR COMPARISON")
    print("-" * 40)
    
    # Compare prior beliefs with posterior estimates
    prior_means = model.priors[:, :, 0].mean(axis=0)  # Average across all LLM priors
    
    print("Prior vs Posterior comparison:")
    print(f"{'Parameter':<20} {'Prior Mean':<12} {'Posterior Mean':<15} {'Difference':<12}")
    print("-" * 60)
    
    for i, feature in enumerate(['intercept'] + feature_names):
        param_name = f'beta_{i}'
        if param_name in trace.posterior.data_vars:
            prior_mean = prior_means[i]
            posterior_mean = posterior_summary[feature]['mean']
            diff = posterior_mean - prior_mean
            print(f"{feature:<20} {prior_mean:>10.4f} {posterior_mean:>13.4f} {diff:>10.4f}")
    
    # 9. Save detailed results
    print("\n9. SAVING DETAILED RESULTS")
    print("-" * 40)
    
    results_dir = project_root / "data/results"
    results_dir.mkdir(exist_ok=True)
    
    # Save individual predictions
    results_df.to_csv(results_dir / "individual_predictions.csv")
    
    # Save parameter estimates
    param_df.to_csv(results_dir / "parameter_estimates.csv")
    
    # Save summary statistics
    summary_stats = {
        'model_fit': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r_squared': float(r_squared),
            'coverage_95': float(coverage)
        },
        'data_summary': {
            'total_points': len(X),
            'training_points': len(X_train),
            'test_points': len(X_test),
            'features': feature_names
        },
        'prior_count': len(model.priors)
    }
    
    with open(results_dir / "model_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Results saved to {results_dir}/")
    print("- individual_predictions.csv")
    print("- parameter_estimates.csv") 
    print("- model_summary.json")
    
    # 10. Create visualizations
    print("\n10. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Actual vs Predicted
    axes[0,0].scatter(results_df['Actual_Return'], results_df['Predicted_Mean'], alpha=0.7)
    axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.5)
    axes[0,0].set_xlabel('Actual Return')
    axes[0,0].set_ylabel('Predicted Return')
    axes[0,0].set_title('Actual vs Predicted')
    
    # Residuals
    axes[0,1].scatter(results_df['Predicted_Mean'], results_df['Prediction_Error'], alpha=0.7)
    axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0,1].set_xlabel('Predicted Return')
    axes[0,1].set_ylabel('Residual')
    axes[0,1].set_title('Residual Plot')
    
    # Prediction intervals
    x_pos = range(len(results_df))
    axes[1,0].errorbar(x_pos, results_df['Predicted_Mean'], 
                      yerr=[results_df['Predicted_Mean'] - results_df['Lower_5%'],
                            results_df['Upper_95%'] - results_df['Predicted_Mean']], 
                      fmt='o', capsize=5, alpha=0.7, label='Predictions')
    axes[1,0].scatter(x_pos, results_df['Actual_Return'], color='red', alpha=0.8, label='Actual')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Return')
    axes[1,0].set_title('Predictions with 95% Intervals')
    axes[1,0].legend()
    
    # Parameter estimates
    param_names = list(posterior_summary.keys())
    param_means = [posterior_summary[p]['mean'] for p in param_names]
    param_errors = [posterior_summary[p]['std'] for p in param_names]
    
    axes[1,1].barh(param_names, param_means, xerr=param_errors, alpha=0.7)
    axes[1,1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1,1].set_xlabel('Parameter Estimate')
    axes[1,1].set_title('Parameter Estimates (±1 std)')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'model_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Diagnostic plots saved to {results_dir}/model_diagnostics.png")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    comprehensive_analysis() 
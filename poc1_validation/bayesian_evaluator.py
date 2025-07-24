"""
Bayesian evaluator for both LLM-elicited and uninformed priors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from base_evaluator import BaseEvaluator
from hanwha_predictor.models.bayesian import HanwhaBayesianModel
from distribution_visualizer import DistributionVisualizer
import json


class BayesianEvaluator(BaseEvaluator):
    """
    Evaluator for Bayesian methods (both elicited and uninformed priors)
    """
    
    def __init__(self, prior_type: str = "elicited", prior_folder: str = "expert_10", include_news: bool = False):
        """
        Initialize Bayesian evaluator
        
        Args:
            prior_type: Either "elicited" or "uninformed"
            prior_folder: Folder name for priors (e.g., "expert_10", "data_informed_3", "data_informed_10", "expert_10_with_news")
            include_news: Whether to use news data (overrides detection from prior_folder name)
        """
        if prior_type not in ["elicited", "uninformed"]:
            raise ValueError("prior_type must be 'elicited' or 'uninformed'")
            
        self.prior_type = prior_type
        self.prior_folder = prior_folder
        
        # Use include_news parameter, or detect from prior folder name if not provided
        if include_news is None:
            include_news = "_with_news" in prior_folder
        
        # Create method name that includes prior folder for better organization
        if prior_type == "elicited":
            method_name = f"bayesian_elicited_{prior_folder}"
        else:
            # For uninformed priors, add _with_news suffix if using news data
            if include_news:
                method_name = "bayesian_uninformed_with_news"
            else:
                method_name = "bayesian_uninformed"
        
        super().__init__(method_name, include_news=include_news)
        
        # Bayesian-specific attributes
        self.model = None
        self.trace = None
        self.posterior_summary = None
        self.visualizer = DistributionVisualizer(self.results_dir)
        
    def train_and_predict(self):
        """Train Bayesian model and generate predictions"""
        print(f"\n🧠 BAYESIAN MODEL TRAINING ({self.prior_type.upper()} PRIORS)")
        if self.prior_type == "elicited":
            print(f"Prior folder: {self.prior_folder}")
        print(f"{'-'*40}")
        
        # Initialize model with custom prior folder
        if self.prior_type == "elicited":
            priors_path = f"data/priors/{self.prior_folder}"
            self.model = HanwhaBayesianModel(priors_dir=priors_path)
            print(f"✓ Using LLM-elicited priors from {self.prior_folder}: {len(self.model.priors)} prior sets")
        else:
            # For uninformed priors, use default path (doesn't matter since we override)
            self.model = HanwhaBayesianModel()
            # Create uninformed priors
            self._create_uninformed_priors()
            print(f"✓ Using uninformed priors: flat distributions")
        
        # Train model
        print("🔄 Training model...")
        self.trace = self.model.train_model(
            self.X_train.values, 
            self.y_train.values, 
            n_samples=1000, 
            n_chains=2
        )
        print("✓ Model training complete")
        
        # Generate predictions
        print("🔮 Generating predictions...")
        predictions = self.model.predict(self.X_test.values)
        print("✓ Predictions generated")
        
        # Create results DataFrame
        self.results_df = pd.DataFrame({
            'Date': self.X_test.index,
            'Actual_Return': self.y_test.values,
            'Predicted_Mean': predictions['mean'],
            'Predicted_Std': predictions['std'],
            'Lower_5%': predictions['quantiles']['5%'],
            'Upper_95%': predictions['quantiles']['95%'],
            'Prediction_Error': self.y_test.values - predictions['mean']
        })
        
        print(f"✓ Results prepared for {len(self.results_df)} test points")
        
    def _create_uninformed_priors(self):
        """Create uninformed (flat) priors for comparison"""
        n_features = len(self.feature_names) + 1  # +1 for intercept
        
        # Create domain-appropriate uninformed priors
        # For financial coefficients, N(0, 1) is more reasonable than N(0, 10)
        # This still represents "uninformed" relative to LLM-elicited priors
        uninformed_priors = np.zeros((1, n_features, 2))  # 1 prior set
        uninformed_priors[0, :, 0] = 0.0  # Mean = 0 (no directional bias)
        
        # Different scales for intercept vs coefficients (common in literature)
        uninformed_priors[0, 0, 1] = 2.0   # Intercept: N(0, 2) - wider for baseline return
        uninformed_priors[0, 1:, 1] = 1.0  # Coefficients: N(0, 1) - reasonable for financial params
        
        # Override the model's priors
        self.model.priors = uninformed_priors
        
        print(f"✓ Created uninformed priors: N(0, 2) for intercept, N(0, 1) for {n_features-1} coefficients")
        
    def analyze_bayesian_diagnostics(self):
        """Perform Bayesian-specific diagnostics"""
        print(f"\n🔍 BAYESIAN DIAGNOSTICS")
        print(f"{'-'*40}")
        
        try:
            import arviz as az
            
            # Convergence diagnostics
            print("⚡ Convergence Diagnostics:")
            rhat = az.rhat(self.trace)
            for var in rhat.data_vars:
                rhat_val = float(rhat[var].values) if rhat[var].values.size == 1 else rhat[var].values.mean()
                status = "✓" if rhat_val < 1.1 else "⚠️"
                print(f"   {status} {var}: R-hat = {rhat_val:.3f}")
            
            # Effective sample size
            print("\n🎯 Effective Sample Size:")
            ess = az.ess(self.trace)
            for var in ess.data_vars:
                ess_val = float(ess[var].values) if ess[var].values.size == 1 else ess[var].values.mean()
                print(f"   ✓ {var}: {ess_val:.0f}")
                
        except ImportError:
            print("⚠️ ArviZ not available for advanced diagnostics")
        
    def analyze_parameters(self):
        """Analyze posterior parameter estimates"""
        print(f"\n🎯 PARAMETER ANALYSIS")
        print(f"{'-'*40}")
        
        # Extract parameter estimates
        self.posterior_summary = {}
        for i, feature in enumerate(['intercept'] + self.feature_names):
            param_name = f'beta_{i}'
            if param_name in self.trace.posterior.data_vars:
                samples = self.trace.posterior[param_name].values.flatten()
                self.posterior_summary[feature] = {
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    'lower_95': np.percentile(samples, 2.5),
                    'upper_95': np.percentile(samples, 97.5)
                }
        
        # Create parameter DataFrame
        param_df = pd.DataFrame(self.posterior_summary).T
        print("✓ Parameter Estimates:")
        print(param_df.round(4))
        
        # Save parameter estimates
        param_df.to_csv(self.results_dir / "parameter_estimates.csv")
        print(f"✓ Parameter estimates saved to parameter_estimates.csv")
        
        return param_df
    
    def save_prior_posterior_comparison(self):
        """Save prior vs posterior comparison data"""
        if not self.posterior_summary:
            return
            
        comparison_data = []
        
        if self.prior_type == "elicited":
            # Use actual elicited priors
            prior_means = self.model.priors[:, :, 0].mean(axis=0)
            for i, feature in enumerate(['intercept'] + self.feature_names):
                if feature in self.posterior_summary:
                    comparison_data.append({
                        'parameter': feature,
                        'prior_mean': prior_means[i],
                        'posterior_mean': self.posterior_summary[feature]['mean'],
                        'posterior_std': self.posterior_summary[feature]['std'],
                        'difference': self.posterior_summary[feature]['mean'] - prior_means[i]
                    })
        else:
            # Use uninformed priors (N(0, 2) for intercept, N(0, 1) for coefficients)
            for feature in ['intercept'] + self.feature_names:
                if feature in self.posterior_summary:
                    comparison_data.append({
                        'parameter': feature,
                        'prior_mean': 0.0,  # All uninformed prior means are 0
                        'posterior_mean': self.posterior_summary[feature]['mean'],
                        'posterior_std': self.posterior_summary[feature]['std'],
                        'difference': self.posterior_summary[feature]['mean'] - 0.0
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.results_dir / "prior_posterior_comparison.csv", index=False)
        print(f"✓ Prior-posterior comparison saved to prior_posterior_comparison.csv")
        
    def compare_prior_posterior(self):
        """Compare prior beliefs with posterior estimates"""
        print(f"\n🔄 PRIOR vs POSTERIOR COMPARISON")
        print(f"{'-'*40}")
        
        if self.prior_type == "elicited":
            # Compare with LLM-elicited priors
            prior_means = self.model.priors[:, :, 0].mean(axis=0)
            
            print(f"{'Parameter':<20} {'Prior Mean':<12} {'Posterior Mean':<15} {'Difference':<12}")
            print("-" * 60)
            
            for i, feature in enumerate(['intercept'] + self.feature_names):
                if feature in self.posterior_summary:
                    prior_mean = prior_means[i]
                    posterior_mean = self.posterior_summary[feature]['mean']
                    diff = posterior_mean - prior_mean
                    print(f"{feature:<20} {prior_mean:>10.4f} {posterior_mean:>13.4f} {diff:>10.4f}")
                    
        else:
            # Compare with uninformed priors (N(0, 2) for intercept, N(0, 1) for coefficients)
            print("Uninformed Priors: N(0, 2) for intercept, N(0, 1) for coefficients")
            print(f"{'Parameter':<20} {'Prior Mean':<12} {'Posterior Mean':<15} {'Difference':<12}")
            print("-" * 60)
            
            for feature in ['intercept'] + self.feature_names:
                if feature in self.posterior_summary:
                    prior_mean = 0.0  # All uninformed prior means are 0
                    posterior_mean = self.posterior_summary[feature]['mean']
                    diff = posterior_mean - prior_mean
                    print(f"{feature:<20} {prior_mean:>10.4f} {posterior_mean:>13.4f} {diff:>10.4f}")
                    
            print("✓ Shows how much data moved us away from uninformed beliefs")
            
    def create_bayesian_plots(self):
        """Create Bayesian-specific plots"""
        print(f"\n🎨 CREATING BAYESIAN PLOTS")
        print(f"{'-'*40}")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residual analysis
        residuals = self.results_df['Prediction_Error']
        axes[0,0].scatter(self.results_df['Predicted_Mean'], residuals, alpha=0.7)
        axes[0,0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0,0].set_xlabel('Predicted Return')
        axes[0,0].set_ylabel('Residual')
        axes[0,0].set_title('Residual Analysis')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Prediction intervals
        x_pos = range(len(self.results_df))
        axes[0,1].errorbar(x_pos, self.results_df['Predicted_Mean'], 
                          yerr=[self.results_df['Predicted_Mean'] - self.results_df['Lower_5%'],
                                self.results_df['Upper_95%'] - self.results_df['Predicted_Mean']], 
                          fmt='o', capsize=5, alpha=0.7, label='Predictions')
        axes[0,1].scatter(x_pos, self.results_df['Actual_Return'], 
                         color='red', alpha=0.8, label='Actual')
        axes[0,1].set_xlabel('Month')
        axes[0,1].set_ylabel('Return')
        axes[0,1].set_title('Predictions with 95% Confidence Intervals')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Parameter estimates
        if self.posterior_summary:
            param_names = list(self.posterior_summary.keys())
            param_means = [self.posterior_summary[p]['mean'] for p in param_names]
            param_errors = [self.posterior_summary[p]['std'] for p in param_names]
            
            axes[1,0].barh(param_names, param_means, xerr=param_errors, alpha=0.7)
            axes[1,0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
            axes[1,0].set_xlabel('Parameter Estimate')
            axes[1,0].set_title('Parameter Estimates (±1 std)')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Prior vs Posterior comparison
        if self.posterior_summary:
            if self.prior_type == "elicited":
                prior_means = self.model.priors[:, :, 0].mean(axis=0)
                posterior_means = [self.posterior_summary[p]['mean'] 
                                 for p in ['intercept'] + self.feature_names]
                
                axes[1,1].scatter(prior_means, posterior_means, alpha=0.7, s=60)
                
                # Perfect agreement line
                min_val = min(min(prior_means), min(posterior_means))
                max_val = max(max(prior_means), max(posterior_means))
                axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                axes[1,1].set_xlabel('Prior Mean')
                axes[1,1].set_ylabel('Posterior Mean')
                axes[1,1].set_title('Prior vs Posterior Beliefs (Elicited)')
                axes[1,1].grid(True, alpha=0.3)
                
            else:
                # Uninformed priors: show how much we moved from N(0, σ) where σ varies by parameter
                prior_means = [0.0] * len(self.feature_names + ['intercept'])  # All uninformed prior means are 0
                posterior_means = [self.posterior_summary[p]['mean'] 
                                 for p in ['intercept'] + self.feature_names]
                
                axes[1,1].scatter(prior_means, posterior_means, alpha=0.7, s=60, color='orange')
                
                # Perfect agreement line
                min_val = min(min(prior_means), min(posterior_means))
                max_val = max(max(prior_means), max(posterior_means))
                axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                # Highlight the uninformed prior at (0,0)
                axes[1,1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
                axes[1,1].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
                
                axes[1,1].set_xlabel('Prior Mean (Uninformed = 0)')
                axes[1,1].set_ylabel('Posterior Mean')
                axes[1,1].set_title('Prior vs Posterior Beliefs (Uninformed)')
                axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'No posterior estimates available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Prior vs Posterior Beliefs')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'bayesian_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"✓ Bayesian plots saved to {self.results_dir}/bayesian_diagnostics.png")
    
    def create_distribution_plots(self):
        """Create detailed prior and posterior distribution plots"""
        print(f"\n🎨 CREATING DISTRIBUTION PLOTS")
        print(f"{'-'*40}")
        
        if not self.posterior_summary:
            print("❌ No posterior summary available")
            return
        
        # 1. Parameter distributions (prior vs posterior)
        self.visualizer.plot_parameter_distributions(
            self.model, self.trace, self.posterior_summary, 
            self.feature_names, self.prior_type
        )
        
        # 2. Prior-posterior comparison
        self.visualizer.plot_prior_posterior_comparison(
            self.model, self.posterior_summary, 
            self.feature_names, self.prior_type
        )
        
        # 3. Mixture components (only for elicited priors)
        if self.prior_type == "elicited":
            print("🔍 Showing mixture components for key parameters...")
            # Show mixture for intercept and most important feature
            self.visualizer.plot_mixture_components(
                self.model, param_idx=0, param_name="intercept"
            )
            
            # Show mixture for first feature
            if len(self.feature_names) > 0:
                self.visualizer.plot_mixture_components(
                    self.model, param_idx=1, param_name=self.feature_names[0]
                )
        
        print("✓ Distribution plots completed")
        
    def run_evaluation(self):
        """Run complete Bayesian evaluation"""
        # Run base evaluation
        metrics = super().run_evaluation()
        
        # Add Bayesian-specific analysis
        self.analyze_bayesian_diagnostics()
        param_df = self.analyze_parameters()
        self.compare_prior_posterior()
        self.save_prior_posterior_comparison()
        self.create_bayesian_plots()
        self.create_distribution_plots()
        
        # Update summary with Bayesian-specific info
        bayesian_summary = {
            'prior_type': self.prior_type,
            'prior_folder': self.prior_folder,
            'n_priors': len(self.model.priors) if self.prior_type == "elicited" else 1,
            'n_parameters': len(self.posterior_summary) if self.posterior_summary else 0
        }
        
        # Save updated summary
        with open(self.results_dir / "summary.json", "r") as f:
            summary = json.load(f)
        summary['bayesian_info'] = bayesian_summary
        
        with open(self.results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return metrics 
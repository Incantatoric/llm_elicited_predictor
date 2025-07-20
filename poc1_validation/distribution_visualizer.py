"""
Distribution visualizer for Bayesian priors and posteriors
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import pandas as pd


class DistributionVisualizer:
    """
    Visualize prior and posterior distributions for Bayesian analysis
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_mixture_pdf(self, means: np.ndarray, stds: np.ndarray, 
                          weights: np.ndarray, x_range: np.ndarray) -> np.ndarray:
        """Create PDF for mixture of normals"""
        pdf = np.zeros_like(x_range)
        
        for i in range(len(means)):
            component_pdf = weights[i] * stats.norm.pdf(x_range, means[i], stds[i])
            pdf += component_pdf
            
        return pdf
    
    def plot_parameter_distributions(self, model, trace, posterior_summary, 
                                   feature_names, prior_type="elicited"):
        """Plot prior and posterior distributions for each parameter"""
        
        n_params = len(feature_names) + 1  # +1 for intercept
        param_names = ['intercept'] + feature_names
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # X-range for plotting
        x_range = np.linspace(-3, 3, 1000)
        
        for i, param_name in enumerate(param_names):
            if i >= 6:  # Only plot first 6 parameters
                break
                
            ax = axes[i]
            
            # Get posterior samples
            param_key = f'beta_{i}'
            if param_key in trace.posterior.data_vars:
                posterior_samples = trace.posterior[param_key].values.flatten()
                
                # Plot posterior
                ax.hist(posterior_samples, bins=50, alpha=0.7, density=True, 
                       color='blue', label='Posterior')
                
                # Plot posterior statistics
                post_mean = posterior_summary[param_name]['mean']
                post_std = posterior_summary[param_name]['std']
                
                # Fitted normal to posterior
                posterior_pdf = stats.norm.pdf(x_range, post_mean, post_std)
                ax.plot(x_range, posterior_pdf, 'b-', linewidth=2, alpha=0.8, 
                       label=f'Posterior N({post_mean:.3f}, {post_std:.3f})')
                
                if prior_type == "elicited":
                    # Plot mixture prior
                    means = model.priors[:, i, 0]
                    stds = model.priors[:, i, 1]
                    weights = np.ones(len(means)) / len(means)
                    
                    mixture_pdf = self.create_mixture_pdf(means, stds, weights, x_range)
                    ax.plot(x_range, mixture_pdf, 'r-', linewidth=2, alpha=0.8,
                           label=f'Mixture Prior ({len(means)} components)')
                    
                    # Show individual components (first few)
                    for j in range(min(5, len(means))):
                        component_pdf = weights[j] * stats.norm.pdf(x_range, means[j], stds[j])
                        ax.plot(x_range, component_pdf, 'r--', alpha=0.3, linewidth=1)
                
                else:
                    # Plot uninformed prior
                    if i == 0:  # Intercept
                        prior_std = 2.0
                    else:  # Coefficients
                        prior_std = 1.0
                    
                    prior_pdf = stats.norm.pdf(x_range, 0, prior_std)
                    ax.plot(x_range, prior_pdf, 'g-', linewidth=2, alpha=0.8,
                           label=f'Uninformed Prior N(0, {prior_std})')
            
            ax.set_title(f'{param_name}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'parameter_distributions_{prior_type}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Parameter distributions saved to parameter_distributions_{prior_type}.png")
    
    def plot_prior_posterior_comparison(self, model, posterior_summary, 
                                      feature_names, prior_type="elicited"):
        """Create comparison plots showing prior vs posterior for each parameter"""
        
        param_names = ['intercept'] + feature_names
        n_params = len(param_names)
        
        # Create a summary comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Prior vs Posterior Means
        if prior_type == "elicited":
            prior_means = model.priors[:, :, 0].mean(axis=0)
            prior_stds = model.priors[:, :, 0].std(axis=0)  # Variation across mixtures
        else:
            prior_means = np.zeros(n_params)
            prior_stds = np.array([2.0] + [1.0] * (n_params - 1))
        
        posterior_means = [posterior_summary[p]['mean'] for p in param_names]
        posterior_stds = [posterior_summary[p]['std'] for p in param_names]
        
        # Scatter plot: Prior vs Posterior means
        axes[0,0].scatter(prior_means, posterior_means, alpha=0.7, s=60)
        
        # Perfect agreement line
        min_val = min(min(prior_means), min(posterior_means))
        max_val = max(max(prior_means), max(posterior_means))
        axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        axes[0,0].set_xlabel('Prior Mean')
        axes[0,0].set_ylabel('Posterior Mean')
        axes[0,0].set_title('Prior vs Posterior Means')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Uncertainty comparison
        axes[0,1].scatter(prior_stds, posterior_stds, alpha=0.7, s=60)
        axes[0,1].set_xlabel('Prior Std')
        axes[0,1].set_ylabel('Posterior Std')
        axes[0,1].set_title('Prior vs Posterior Uncertainty')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Parameter-wise comparison
        x_pos = np.arange(len(param_names))
        width = 0.35
        
        axes[1,0].bar(x_pos - width/2, prior_means, width, alpha=0.7, 
                     label='Prior Mean', color='red')
        axes[1,0].bar(x_pos + width/2, posterior_means, width, alpha=0.7, 
                     label='Posterior Mean', color='blue')
        
        axes[1,0].set_xlabel('Parameter')
        axes[1,0].set_ylabel('Mean Value')
        axes[1,0].set_title('Parameter Means Comparison')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(param_names, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Shrinkage analysis
        differences = np.array(posterior_means) - np.array(prior_means)
        axes[1,1].bar(x_pos, differences, alpha=0.7, color='green')
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,1].set_xlabel('Parameter')
        axes[1,1].set_ylabel('Posterior - Prior')
        axes[1,1].set_title('Learning from Data (Shrinkage)')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(param_names, rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'prior_posterior_comparison_{prior_type}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Comparison plots saved to prior_posterior_comparison_{prior_type}.png")
    
    def plot_mixture_components(self, model, param_idx=0, param_name="intercept"):
        """Plot individual components of mixture prior"""
        
        if hasattr(model, 'priors'):
            means = model.priors[:, param_idx, 0]
            stds = model.priors[:, param_idx, 1]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot individual components
            x_range = np.linspace(-3, 3, 1000)
            
            for i in range(min(20, len(means))):  # Show first 20 components
                pdf = stats.norm.pdf(x_range, means[i], stds[i])
                ax1.plot(x_range, pdf, alpha=0.3, linewidth=1)
            
            # Plot mixture
            weights = np.ones(len(means)) / len(means)
            mixture_pdf = self.create_mixture_pdf(means, stds, weights, x_range)
            ax1.plot(x_range, mixture_pdf, 'r-', linewidth=3, label='Mixture')
            
            ax1.set_title(f'Mixture Components for {param_name}')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Show distribution of component means and stds
            ax2.scatter(means, stds, alpha=0.6)
            ax2.set_xlabel('Component Mean')
            ax2.set_ylabel('Component Std')
            ax2.set_title(f'Component Parameters for {param_name}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'mixture_components_{param_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✓ Mixture components plot saved to mixture_components_{param_name}.png") 
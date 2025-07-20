import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class HanwhaInterpreter:
    """
    Interpretability tools for 한화솔루션 stock prediction model
    Focus on business-friendly explanations for your boss
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        
    def plot_prior_distributions(self, priors: np.ndarray, save_path: str = None):
        """
        Visualize the mixture of elicited priors
        Shows what the LLM believes about each parameter
        """
        n_features = len(self.feature_names)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot intercept
        intercept_means = priors[:, 0, 0]
        axes[0].hist(intercept_means, bins=20, alpha=0.7, color='gray')
        axes[0].set_title('Intercept Prior')
        axes[0].set_xlabel('Mean Value')
        axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # Plot each feature
        for i, feature_name in enumerate(self.feature_names):
            feature_means = priors[:, i+1, 0]  # +1 because intercept is at index 0
            
            axes[i+1].hist(feature_means, bins=20, alpha=0.7)
            axes[i+1].set_title(f'{feature_name}\nLLM Prior Beliefs')
            axes[i+1].set_xlabel('Mean Value')
            axes[i+1].axvline(0, color='red', linestyle='--', alpha=0.5, label='No effect')
            
            # Add interpretation
            mean_belief = np.mean(feature_means)
            confidence = 1 / np.std(feature_means) if np.std(feature_means) > 0 else float('inf')
            
            axes[i+1].text(0.05, 0.95, 
                          f'Avg belief: {mean_belief:.3f}\n'
                          f'Confidence: {"High" if confidence > 2 else "Medium" if confidence > 1 else "Low"}',
                          transform=axes[i+1].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def explain_parameter_effects(self, trace, priors: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Generate business-friendly explanations of parameter effects
        """
        explanations = {}
        
        # Parameter interpretations
        parameter_meanings = {
            'kospi_return': 'Korean stock market performance',
            'oil_price_change': 'Oil price changes (input costs for Hanwha)',
            'usd_krw_change': 'Exchange rate changes (export competitiveness)',
            'vix_change': 'Market volatility and risk sentiment',
            'materials_sector_return': 'Chemical sector performance'
        }
        
        for i, feature_name in enumerate(self.feature_names):
            # Get posterior samples for this parameter
            posterior_samples = trace.posterior[f'beta_{i+1}'].values.flatten()
            
            # Get prior beliefs
            prior_means = priors[:, i+1, 0]
            
            # Calculate statistics
            posterior_mean = np.mean(posterior_samples)
            posterior_std = np.std(posterior_samples)
            prior_mean = np.mean(prior_means)
            
            # Determine effect size and direction
            effect_size = abs(posterior_mean)
            direction = "positive" if posterior_mean > 0 else "negative"
            confidence = "high" if posterior_std < 0.1 else "medium" if posterior_std < 0.2 else "low"
            
            # Generate explanation
            explanation = {
                'feature': feature_name,
                'meaning': parameter_meanings.get(feature_name, feature_name),
                'posterior_mean': posterior_mean,
                'posterior_std': posterior_std,
                'prior_belief': prior_mean,
                'direction': direction,
                'effect_size': effect_size,
                'confidence': confidence,
                'business_interpretation': self._generate_business_interpretation(
                    feature_name, posterior_mean, confidence, parameter_meanings
                )
            }
            
            explanations[feature_name] = explanation
        
        return explanations
    
    def _generate_business_interpretation(self, feature_name: str, effect: float, 
                                        confidence: str, meanings: Dict[str, str]) -> str:
        """Generate business-friendly interpretation"""
        
        meaning = meanings.get(feature_name, feature_name)
        direction = "increases" if effect > 0 else "decreases"
        magnitude = "significantly" if abs(effect) > 0.2 else "moderately" if abs(effect) > 0.1 else "slightly"
        
        interpretation = f"When {meaning} goes up by 1 standard deviation, "
        interpretation += f"Hanwha Solutions returns {direction} {magnitude} "
        interpretation += f"(effect size: {effect:.3f}). "
        interpretation += f"We have {confidence} confidence in this relationship."
        
        return interpretation
    
    def scenario_analysis(self, model, trace, feature_names: List[str], 
                         scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Perform what-if scenario analysis for your boss
        """
        results = []
        
        for scenario_name, scenario_values in scenarios.items():
            # Create feature vector for this scenario
            X_scenario = np.array([[scenario_values.get(f, 0.0) for f in feature_names]])
            
            # Make prediction
            with model:
                posterior_predictive = pm.sample_posterior_predictive(
                    trace, predictions=True, extend_inferencedata=False
                )
            
            prediction_samples = posterior_predictive['y_obs'][0]
            
            result = {
                'scenario': scenario_name,
                'expected_return': np.mean(prediction_samples),
                'std_dev': np.std(prediction_samples),
                'prob_positive': (prediction_samples > 0).mean(),
                'var_95': np.percentile(prediction_samples, 5),  # 95% VaR
                'scenario_inputs': scenario_values
            }
            
            results.append(result)
        
        return pd.DataFrame(results)

# Example scenarios for your boss
BUSINESS_SCENARIOS = {
    "Bull Market": {
        "kospi_return": 0.1,          # KOSPI up 10%
        "oil_price_change": -0.05,    # Oil down 5% (good for costs)
        "usd_krw_change": 0.05,       # KRW weakens (good for exports)
        "vix_change": -0.2,           # Low volatility
        "materials_sector_return": 0.08
    },
    "Bear Market": {
        "kospi_return": -0.1,         # KOSPI down 10%
        "oil_price_change": 0.1,      # Oil up 10% (bad for costs)
        "usd_krw_change": -0.05,      # KRW strengthens (bad for exports)
        "vix_change": 0.3,            # High volatility
        "materials_sector_return": -0.08
    },
    "Oil Shock": {
        "kospi_return": 0.0,          # Neutral market
        "oil_price_change": 0.2,      # Oil spikes 20%
        "usd_krw_change": 0.0,
        "vix_change": 0.1,
        "materials_sector_return": 0.0
    },
    "Export Boost": {
        "kospi_return": 0.05,
        "oil_price_change": 0.0,
        "usd_krw_change": 0.1,        # Weak KRW helps exports
        "vix_change": 0.0,
        "materials_sector_return": 0.05
    }
}
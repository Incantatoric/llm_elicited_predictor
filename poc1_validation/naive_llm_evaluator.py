#!/usr/bin/env python3
"""
Naive LLM Evaluator - Direct prediction using in-context learning
Implements the baseline approach from Capstick et al. paper
"""

from base_evaluator import BaseEvaluator
import numpy as np
import pandas as pd
import json
import openai
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NaiveLLMEvaluator(BaseEvaluator):
    """
    Naive LLM evaluator using direct in-context prediction
    This represents the baseline approach in the Capstick paper
    """
    
    def __init__(self):
        super().__init__("naive_llm")  # Just pass method name, BaseEvaluator handles the path
        
        # LLM-specific attributes
        self.client = openai.OpenAI()  # Will use OPENAI_API_KEY from env
        self.model = "gpt-3.5-turbo"
        self.predictions = None
        self.prediction_samples = None
        
        # BaseEvaluator already creates results_dir, no need to create again
    
    def train_and_predict(self):
        """Generate predictions using in-context learning"""
        print(f"🤖 GENERATING LLM PREDICTIONS")
        print(f"{'-'*40}")
        
        # Prepare context from training data
        context = self._prepare_context()
        
        # Generate predictions for test set
        self.predictions, self.prediction_samples = self._generate_predictions(context)
        
        print(f"✓ Generated {len(self.predictions)} predictions")
        
        # Create results DataFrame in BaseEvaluator format
        self._create_results_dataframe()
        print(f"✓ Results formatted for evaluation")
    
    def _prepare_context(self) -> str:
        """Prepare training data as context for LLM"""
        
        # Create a sample of training data for context (limit to avoid token limits)
        n_context = min(50, len(self.X_train))  # Use up to 50 examples
        indices = np.random.choice(len(self.X_train), n_context, replace=False)
        
        # Handle DataFrame indexing properly
        context_X = self.X_train.iloc[indices]  # Use .iloc for integer indexing
        context_y = self.y_train.iloc[indices]  # Use .iloc for integer indexing
        
        # Format as text
        context_lines = []
        context_lines.append("Historical Data (Features -> Return):")
        context_lines.append("=" * 50)
        
        for i in range(len(context_X)):
            feature_str = ", ".join([f"{name}={val:.3f}" for name, val in 
                                   zip(self.feature_names, context_X.iloc[i])])
            context_lines.append(f"{feature_str} -> Return: {context_y.iloc[i]:.4f}")
        
        return "\n".join(context_lines)
    
    def _generate_predictions(self, context: str) -> tuple:
        """Generate predictions using LLM"""
        
        predictions = []
        all_samples = []
        
        # Generate multiple samples for uncertainty quantification
        n_samples = 10  # Generate multiple predictions per test point
        
        for i in range(len(self.X_test)):
            print(f"Predicting for test sample {i+1}/{len(self.X_test)}")
            
            # Format test features  
            test_features = self.X_test.iloc[i]
            feature_str = ", ".join([f"{name}={val:.3f}" for name, val in 
                                   zip(self.feature_names, test_features)])
            
            # Generate multiple samples for this test point
            samples = []
            for sample_idx in range(n_samples):
                prediction = self._get_single_prediction(context, feature_str, sample_idx)
                if prediction is not None:
                    samples.append(prediction)
            
            if len(samples) > 0:
                # Use mean as point prediction
                predictions.append(np.mean(samples))
                all_samples.append(samples)
            else:
                # Fallback if all predictions failed
                predictions.append(0.0)
                all_samples.append([0.0])
        
        return np.array(predictions), all_samples
    
    def _create_results_dataframe(self):
        """Create results DataFrame in BaseEvaluator format"""
        if self.predictions is None or self.y_test is None:
            raise ValueError("Predictions and test data must be available")
        
        # Create basic DataFrame (matching BayesianEvaluator format)
        self.results_df = pd.DataFrame({
            'Date': self.X_test.index,  # Add Date column for plotting
            'Actual_Return': self.y_test.values if hasattr(self.y_test, 'values') else self.y_test,
            'Predicted_Mean': self.predictions,
            'Prediction_Error': (self.y_test.values if hasattr(self.y_test, 'values') else self.y_test) - self.predictions
        })
        
        # Add prediction intervals if we have samples
        if self.prediction_samples and len(self.prediction_samples[0]) > 1:
            lower_bounds = []
            upper_bounds = []
            
            for samples in self.prediction_samples:
                lower_bounds.append(np.percentile(samples, 5))
                upper_bounds.append(np.percentile(samples, 95))
            
            self.results_df['Lower_5%'] = lower_bounds
            self.results_df['Upper_95%'] = upper_bounds
    
    def _get_single_prediction(self, context: str, feature_str: str, sample_idx: int) -> float:
        """Get a single prediction from LLM"""
        
        prompt = f"""You are a financial analyst predicting stock returns for 한화솔루션.

{context}

Based on this historical pattern, predict the return for:
{feature_str}

Instructions:
- Analyze the relationship between features and returns
- Consider the patterns in the historical data
- Respond with ONLY a numerical value (e.g., 0.0234 for 2.34% return)
- Be precise to 4 decimal places
- Negative values indicate losses, positive indicate gains

Prediction:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Some randomness for sampling
                max_tokens=50
            )
            
            # Extract numerical value
            prediction_text = response.choices[0].message.content.strip()
            
            # Try to extract number from response
            import re
            numbers = re.findall(r'-?\d+\.?\d*', prediction_text)
            
            if numbers:
                return float(numbers[0])
            else:
                logger.warning(f"Could not extract number from: {prediction_text}")
                return None
                
        except Exception as e:
            logger.error(f"LLM prediction failed (sample {sample_idx}): {e}")
            return None
    
    def calculate_uncertainty_metrics(self):
        """Calculate uncertainty-related metrics from prediction samples"""
        if not self.prediction_samples:
            return {}
        
        # Calculate prediction intervals
        lower_bounds = []
        upper_bounds = []
        prediction_stds = []
        
        for samples in self.prediction_samples:
            if len(samples) > 1:
                lower_bounds.append(np.percentile(samples, 5))
                upper_bounds.append(np.percentile(samples, 95))
                prediction_stds.append(np.std(samples))
            else:
                # Single sample case
                lower_bounds.append(samples[0])
                upper_bounds.append(samples[0])
                prediction_stds.append(0.0)
        
        # Calculate coverage (what % of actual values fall within prediction intervals)
        if len(lower_bounds) > 0:
            y_test_vals = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
            coverage_90 = np.mean((y_test_vals >= lower_bounds) & (y_test_vals <= upper_bounds))
        else:
            coverage_90 = 0.0
        
        return {
            'prediction_std_mean': np.mean(prediction_stds),
            'prediction_std_std': np.std(prediction_stds),
            'coverage_90': coverage_90,
            'avg_prediction_interval_width': np.mean(np.array(upper_bounds) - np.array(lower_bounds))
        }
    
    def analyze_parameters(self):
        """Analyze LLM prediction patterns (no parameters like Bayesian)"""
        print(f"\n📊 ANALYZING LLM PREDICTIONS")
        print(f"{'-'*40}")
        
        if self.predictions is None:
            print("❌ No predictions available for analysis")
            return
        
        # Basic prediction statistics
        pred_stats = {
            'mean_prediction': float(np.mean(self.predictions)),
            'std_prediction': float(np.std(self.predictions)),
            'min_prediction': float(np.min(self.predictions)),
            'max_prediction': float(np.max(self.predictions)),
            'n_positive': int(np.sum(self.predictions > 0)),
            'n_negative': int(np.sum(self.predictions < 0)),
            'prediction_range': float(np.max(self.predictions) - np.min(self.predictions))
        }
        
        # Print analysis
        print(f"Mean Prediction: {pred_stats['mean_prediction']:.4f}")
        print(f"Prediction Std: {pred_stats['std_prediction']:.4f}")
        print(f"Prediction Range: [{pred_stats['min_prediction']:.4f}, {pred_stats['max_prediction']:.4f}]")
        print(f"Positive Predictions: {pred_stats['n_positive']}/{len(self.predictions)}")
        print(f"Negative Predictions: {pred_stats['n_negative']}/{len(self.predictions)}")
        
        # Add uncertainty metrics
        uncertainty_metrics = self.calculate_uncertainty_metrics()
        pred_stats.update(uncertainty_metrics)
        
        if uncertainty_metrics:
            print(f"Average Prediction Std: {uncertainty_metrics.get('prediction_std_mean', 0):.4f}")
            print(f"90% Coverage: {uncertainty_metrics.get('coverage_90', 0):.2%}")
        
        # Save analysis
        with open(self.results_dir / "prediction_analysis.json", 'w') as f:
            json.dump(pred_stats, f, indent=2)
        
        self.parameter_summary = pred_stats
        print(f"✓ Prediction analysis saved to prediction_analysis.json")
    
    def create_model_plots(self):
        """Create plots specific to LLM predictions"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print(f"\n📈 CREATING LLM PLOTS")
        print(f"{'-'*40}")
        
        if self.predictions is None or self.y_test is None:
            print("❌ No data available for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Naive LLM Prediction Analysis', fontsize=16)
        
        # 1. Predictions vs Actual
        axes[0, 0].scatter(self.y_test, self.predictions, alpha=0.6)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Actual Returns')
        axes[0, 0].set_ylabel('Predicted Returns')
        axes[0, 0].set_title('Predictions vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = self.predictions - self.y_test
        axes[0, 1].scatter(self.predictions, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Predicted Returns')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction Distribution
        axes[1, 0].hist(self.predictions, bins=20, alpha=0.7, density=True, label='Predictions')
        axes[1, 0].hist(self.y_test, bins=20, alpha=0.7, density=True, label='Actual')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Uncertainty (if available)
        if self.prediction_samples and len(self.prediction_samples[0]) > 1:
            uncertainties = [np.std(samples) for samples in self.prediction_samples]
            axes[1, 1].scatter(self.predictions, uncertainties, alpha=0.6)
            axes[1, 1].set_xlabel('Mean Prediction')
            axes[1, 1].set_ylabel('Prediction Std')
            axes[1, 1].set_title('Prediction Uncertainty')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Uncertainty Data\n(Single Sample)', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Prediction Uncertainty')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'llm_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ LLM plots saved to llm_predictions.png")
    
    def run_evaluation(self):
        """Run complete naive LLM evaluation"""
        print(f"\n{'='*60}")
        print(f"NAIVE LLM EVALUATION")
        print(f"{'='*60}")
        
        # Load data and split
        self.load_data()
        self.create_train_test_split()
        
        # Generate predictions
        self.train_and_predict()
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Analysis and visualization
        self.analyze_parameters()
        self.create_basic_plots()
        self.create_model_plots()
        
        # Create summary
        summary = {
            'method': 'naive_llm',
            'model': self.model,
            'n_train': len(self.X_train),
            'n_test': len(self.X_test),
            'metrics': self.metrics,
            'prediction_stats': getattr(self, 'parameter_summary', {}),
            'feature_names': self.feature_names
        }
        
        # Save summary
        with open(self.results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ NAIVE LLM EVALUATION COMPLETE")
        print(f"Results saved to: {self.results_dir}")
        
        return self.metrics 
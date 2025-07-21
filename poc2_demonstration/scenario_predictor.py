#!/usr/bin/env python3
"""
Scenario-Based Prediction Engine for PoC 2: Executive Decision Support
Uses trained Bayesian models to generate predictions for executive scenarios
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our components
from scenario_manager import ScenarioManager, Scenario
import sys
sys.path.append(str(Path(__file__).parent.parent))
from hanwha_predictor.models.bayesian import HanwhaBayesianModel, create_uninformative_model
import pymc as pm


@dataclass
class ScenarioPrediction:
    """Prediction results for a single scenario"""
    scenario_name: str
    point_prediction: float
    uncertainty_std: float
    confidence_intervals: Dict[str, float]  # '5%', '25%', '75%', '95%'
    probability_positive: float
    risk_assessment: str
    samples: np.ndarray
    
    
@dataclass
class ScenarioAnalysis:
    """Complete analysis results for multiple scenarios"""
    predictions: List[ScenarioPrediction]
    comparative_metrics: Dict
    recommendations: List[str]
    generated_at: datetime


class ScenarioPredictor:
    """
    Scenario-based prediction engine using trained Bayesian models
    
    Professional Features:
    - Multiple model support (elicited vs uninformed priors)
    - Uncertainty quantification with confidence levels
    - Risk assessment and probability calculations
    - Batch scenario processing
    - Executive-friendly output formatting
    """
    
    def __init__(self, model_type: str = "elicited", project_root: str = None):
        """
        Initialize scenario predictor
        
        Args:
            model_type: "elicited" or "uninformed" - which Bayesian model to use
            project_root: Project root directory
        """
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.model_type = model_type
        self.model = None
        self.trace = None
        self.feature_names = None
        
        # Initialize scenario manager
        self.scenario_manager = ScenarioManager(project_root=str(self.project_root))
        
        # Load trained model
        self._load_trained_model()
        
        print(f"✅ ScenarioPredictor initialized ({model_type} priors)")
        print(f"🧠 Model loaded with trained parameters")
    
    def _load_trained_model(self):
        """Load pre-trained Bayesian model from PoC 1 results"""
        try:
            # Load the training data to retrain model (since we need the trace)
            X = pd.read_csv(self.project_root / 'data/processed/features_standardized.csv', index_col=0)
            y = pd.read_csv(self.project_root / 'data/processed/target_returns.csv', index_col=0).squeeze()
            
            with open(self.project_root / 'data/processed/metadata.json') as f:
                metadata = json.load(f)
            self.feature_names = metadata['feature_names']
            
            # Use training data (excluding last 6 months which were test data)
            split_idx = len(X) - 6
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            
            # Initialize and train model
            if self.model_type == "elicited":
                self.model = HanwhaBayesianModel()
                print(f"📊 Using LLM-elicited priors: {len(self.model.priors)} prior sets")
            else:
                # For uninformed, we'll create a simple model
                print(f"📊 Using uninformed priors")
                self.model = self._create_uninformed_model(X_train.values, y_train.values)
            
            # Train the model to get trace
            print("🔄 Training Bayesian model for scenario predictions...")
            if self.model_type == "elicited":
                self.trace = self.model.train_model(
                    X_train.values, y_train.values, 
                    n_samples=2000, n_chains=2
                )
            else:
                # Train uninformed model
                with self.model:
                    self.trace = pm.sample(
                        draws=2000, chains=2, cores=2,
                        return_inferencedata=True, random_seed=42
                    )
            
            print("✅ Model training complete")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}")
    
    def _create_uninformed_model(self, X: np.ndarray, y: np.ndarray):
        """Create uninformed Bayesian model"""
        n_features = X.shape[1]
        
        model = pm.Model()
        with model:
            # Uninformative priors
            intercept = pm.Normal('intercept', mu=0, sigma=2)
            betas = pm.Normal('betas', mu=0, sigma=1, shape=n_features)
            
            # Linear regression
            mu = intercept + pm.math.dot(X, betas)
            sigma = pm.HalfCauchy('sigma', beta=1.0)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        return model
    
    def predict_scenario(self, scenario: Scenario, n_samples: int = 1000) -> ScenarioPrediction:
        """
        Generate prediction for a single scenario
        
        Args:
            scenario: Scenario object from scenario_manager
            n_samples: Number of posterior predictive samples
            
        Returns:
            ScenarioPrediction with complete analysis
        """
        print(f"🔮 Predicting scenario: {scenario.name}")
        
        # Convert scenario to feature vector
        feature_vector = self.scenario_manager.scenario_to_feature_vector(scenario)
        feature_vector = feature_vector.reshape(1, -1)  # Shape for single prediction
        
        # Generate posterior predictive samples
        if self.model_type == "elicited":
            predictions = self.model.predict(feature_vector, n_samples=n_samples)
            samples = predictions['samples'].flatten()
            point_pred = float(predictions['mean'][0])
            pred_std = float(predictions['std'][0])
            
        else:
            # For uninformed model, need to manually do posterior predictive
            samples = self._sample_posterior_predictive_uninformed(feature_vector, n_samples)
            point_pred = float(np.mean(samples))
            pred_std = float(np.std(samples))
        
        # Calculate confidence intervals
        confidence_intervals = {
            '5%': float(np.percentile(samples, 5)),
            '25%': float(np.percentile(samples, 25)),
            '75%': float(np.percentile(samples, 75)), 
            '95%': float(np.percentile(samples, 95))
        }
        
        # Calculate probability of positive return
        prob_positive = float(np.mean(samples > 0))
        
        # Risk assessment based on uncertainty and prediction
        risk_assessment = self._assess_risk(point_pred, pred_std, samples)
        
        return ScenarioPrediction(
            scenario_name=scenario.name,
            point_prediction=point_pred,
            uncertainty_std=pred_std,
            confidence_intervals=confidence_intervals,
            probability_positive=prob_positive,
            risk_assessment=risk_assessment,
            samples=samples
        )
    
    def _sample_posterior_predictive_uninformed(self, X_new: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample posterior predictive for uninformed model"""
        # Extract posterior samples from trace
        intercept_samples = self.trace.posterior['intercept'].values.flatten()
        beta_samples = self.trace.posterior['betas'].values.reshape(-1, X_new.shape[1])
        sigma_samples = self.trace.posterior['sigma'].values.flatten()
        
        # Random sample indices
        n_total_samples = len(intercept_samples)
        sample_indices = np.random.choice(n_total_samples, n_samples, replace=True)
        
        # Generate predictions
        predictions = []
        for i in sample_indices:
            mu = intercept_samples[i] + np.dot(X_new[0], beta_samples[i])
            pred = np.random.normal(mu, sigma_samples[i])
            predictions.append(pred)
            
        return np.array(predictions)
    
    def _assess_risk(self, point_pred: float, std: float, samples: np.ndarray) -> str:
        """Assess risk level based on prediction characteristics"""
        # Calculate coefficient of variation (relative uncertainty)
        cv = abs(std / point_pred) if point_pred != 0 else float('inf')
        
        # Calculate probability of loss
        prob_loss = np.mean(samples < 0)
        
        # Risk classification logic
        if cv > 10.0 or prob_loss > 0.6:
            return "HIGH"
        elif cv > 5.0 or prob_loss > 0.55:
            return "MEDIUM" 
        else:
            return "LOW"
    
    def analyze_scenarios(self, scenarios: List[Scenario], 
                         n_samples: int = 1000) -> ScenarioAnalysis:
        """
        Analyze multiple scenarios and provide comparative insights
        
        Args:
            scenarios: List of scenarios to analyze
            n_samples: Number of samples per scenario
            
        Returns:
            Complete scenario analysis with recommendations
        """
        print(f"\n📊 ANALYZING {len(scenarios)} SCENARIOS")
        print("=" * 50)
        
        # Generate predictions for all scenarios
        predictions = []
        for scenario in scenarios:
            pred = self.predict_scenario(scenario, n_samples)
            predictions.append(pred)
        
        # Calculate comparative metrics
        comparative_metrics = self._calculate_comparative_metrics(predictions)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(predictions, comparative_metrics)
        
        return ScenarioAnalysis(
            predictions=predictions,
            comparative_metrics=comparative_metrics,
            recommendations=recommendations,
            generated_at=datetime.now()
        )
    
    def _calculate_comparative_metrics(self, predictions: List[ScenarioPrediction]) -> Dict:
        """Calculate comparative metrics across scenarios"""
        point_preds = [p.point_prediction for p in predictions]
        uncertainties = [p.uncertainty_std for p in predictions]
        prob_positives = [p.probability_positive for p in predictions]
        
        return {
            'best_case_scenario': max(enumerate(point_preds), key=lambda x: x[1])[0],
            'worst_case_scenario': min(enumerate(point_preds), key=lambda x: x[1])[0],
            'most_certain_scenario': min(enumerate(uncertainties), key=lambda x: x[1])[0],
            'most_uncertain_scenario': max(enumerate(uncertainties), key=lambda x: x[1])[0],
            'prediction_range': {
                'min': float(min(point_preds)),
                'max': float(max(point_preds)),
                'spread': float(max(point_preds) - min(point_preds))
            },
            'average_probability_positive': float(np.mean(prob_positives)),
            'scenarios_with_high_risk': sum(1 for p in predictions if p.risk_assessment == "HIGH")
        }
    
    def _generate_recommendations(self, predictions: List[ScenarioPrediction], 
                                 metrics: Dict) -> List[str]:
        """Generate executive recommendations based on scenario analysis"""
        recommendations = []
        
        # Risk-based recommendations
        high_risk_count = metrics['scenarios_with_high_risk']
        if high_risk_count > len(predictions) / 2:
            recommendations.append(
                f"⚠️ HIGH RISK: {high_risk_count}/{len(predictions)} scenarios show high uncertainty. "
                "Consider risk mitigation strategies."
            )
        
        # Return potential recommendations
        avg_prob_positive = metrics['average_probability_positive']
        if avg_prob_positive < 0.4:
            recommendations.append(
                f"📉 BEARISH OUTLOOK: Average {avg_prob_positive:.1%} chance of positive returns. "
                "Consider defensive positioning."
            )
        elif avg_prob_positive > 0.7:
            recommendations.append(
                f"📈 BULLISH OUTLOOK: Average {avg_prob_positive:.1%} chance of positive returns. "
                "Consider increasing exposure."
            )
        
        # Spread-based recommendations  
        pred_spread = metrics['prediction_range']['spread']
        if pred_spread > 0.3:  # 30% spread
            recommendations.append(
                f"🎯 HIGH SCENARIO SENSITIVITY: {pred_spread:.1%} spread between scenarios. "
                "Focus on scenario monitoring and contingency planning."
            )
        
        return recommendations
    
    def display_scenario_prediction(self, prediction: ScenarioPrediction):
        """Display executive-friendly prediction summary"""
        print(f"\n{'='*60}")
        print(f"🔮 PREDICTION: {prediction.scenario_name}")
        print(f"{'='*60}")
        
        # Main prediction
        print(f"📊 Expected Return: {prediction.point_prediction:+.2%}")
        print(f"📏 Uncertainty (±1σ): {prediction.uncertainty_std:.2%}")
        print(f"🎯 Probability of Gain: {prediction.probability_positive:.1%}")
        
        # Confidence intervals
        print(f"\n📈 Confidence Intervals:")
        ci = prediction.confidence_intervals
        print(f"   90% Range: [{ci['5%']:+.2%}, {ci['95%']:+.2%}]")
        print(f"   50% Range: [{ci['25%']:+.2%}, {ci['75%']:+.2%}]")
        
        # Risk assessment
        risk_colors = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}
        risk_emoji = risk_colors.get(prediction.risk_assessment, '⚪')
        print(f"\n⚖️  Risk Level: {risk_emoji} {prediction.risk_assessment}")
        
        print("-" * 60)
    
    def create_scenario_comparison_plot(self, analysis: ScenarioAnalysis, save_path: str = None):
        """Create executive-friendly comparison plots"""
        predictions = analysis.predictions
        n_scenarios = len(predictions)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📊 Executive Scenario Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Point predictions with confidence intervals
        scenario_names = [p.scenario_name for p in predictions]
        point_preds = [p.point_prediction for p in predictions]
        lower_95 = [p.confidence_intervals['5%'] for p in predictions]
        upper_95 = [p.confidence_intervals['95%'] for p in predictions]
        
        x_pos = np.arange(n_scenarios)
        axes[0, 0].errorbar(x_pos, point_preds, 
                           yerr=[np.array(point_preds) - np.array(lower_95),
                                 np.array(upper_95) - np.array(point_preds)],
                           fmt='o', capsize=8, linewidth=2, markersize=8)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Scenario')
        axes[0, 0].set_ylabel('Expected Return (%)')
        axes[0, 0].set_title('📈 Expected Returns with 90% Confidence Intervals')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Risk assessment
        risk_levels = [p.risk_assessment for p in predictions]
        risk_counts = pd.Series(risk_levels).value_counts()
        risk_colors = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
        colors = [risk_colors.get(risk, 'gray') for risk in risk_counts.index]
        
        axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, colors=colors,
                       autopct='%1.0f%%', startangle=90)
        axes[0, 1].set_title('⚖️ Risk Level Distribution')
        
        # 3. Probability of positive returns
        prob_positive = [p.probability_positive for p in predictions]
        bars = axes[1, 0].bar(x_pos, prob_positive, color='lightblue', edgecolor='navy')
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Threshold')
        axes[1, 0].set_xlabel('Scenario')
        axes[1, 0].set_ylabel('Probability (%)')
        axes[1, 0].set_title('🎯 Probability of Positive Returns')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, prob_positive):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{prob:.0%}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Uncertainty comparison
        uncertainties = [p.uncertainty_std for p in predictions]
        axes[1, 1].bar(x_pos, uncertainties, color='lightcoral', edgecolor='darkred')
        axes[1, 1].set_xlabel('Scenario')
        axes[1, 1].set_ylabel('Uncertainty (σ)')
        axes[1, 1].set_title('📏 Prediction Uncertainty')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Executive dashboard saved: {save_path}")
        
        plt.show()


def demo_scenario_predictor():
    """Demonstrate scenario prediction capabilities"""
    print("🔮 SCENARIO PREDICTION ENGINE DEMONSTRATION")
    print("="*60)
    
    # Initialize predictor with elicited priors
    predictor = ScenarioPredictor(model_type="elicited")
    
    # Get some template scenarios
    scenarios = [
        predictor.scenario_manager.get_template('bull_market'),
        predictor.scenario_manager.get_template('bear_market'),
        predictor.scenario_manager.get_template('baseline'),
        predictor.scenario_manager.get_template('geopolitical_stress')
    ]
    
    # Analyze all scenarios
    analysis = predictor.analyze_scenarios(scenarios, n_samples=500)  # Smaller for demo
    
    # Display individual predictions
    for prediction in analysis.predictions:
        predictor.display_scenario_prediction(prediction)
    
    # Show comparative analysis
    print(f"\n{'📋 COMPARATIVE ANALYSIS'}")
    print("=" * 40)
    
    metrics = analysis.comparative_metrics
    scenarios_names = [p.scenario_name for p in analysis.predictions]
    
    print(f"🏆 Best Case: {scenarios_names[metrics['best_case_scenario']]}")
    print(f"📉 Worst Case: {scenarios_names[metrics['worst_case_scenario']]}")
    print(f"🎯 Most Certain: {scenarios_names[metrics['most_certain_scenario']]}")
    print(f"❓ Most Uncertain: {scenarios_names[metrics['most_uncertain_scenario']]}")
    print(f"📊 Prediction Range: {metrics['prediction_range']['min']:+.2%} to {metrics['prediction_range']['max']:+.2%}")
    print(f"🎲 Avg Probability Positive: {metrics['average_probability_positive']:.1%}")
    
    # Show recommendations
    print(f"\n💼 EXECUTIVE RECOMMENDATIONS:")
    print("-" * 30)
    for i, rec in enumerate(analysis.recommendations, 1):
        print(f"{i}. {rec}")
    
    # Create visualization
    save_path = Path(__file__).parent / "executive_dashboard.png"
    predictor.create_scenario_comparison_plot(analysis, save_path=str(save_path))
    
    print(f"\n✅ Stage 2 demonstration complete!")
    print(f"Next: Stage 3 - Executive Explanation Engine")


if __name__ == "__main__":
    demo_scenario_predictor() 
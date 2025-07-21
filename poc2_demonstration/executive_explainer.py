#!/usr/bin/env python3
"""
Executive Explanation Engine for PoC 2: Decision Support
Converts technical Bayesian outputs into executive-friendly business insights
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
from scenario_manager import ScenarioManager, Scenario, ScenarioCondition
from scenario_predictor import ScenarioPredictor, ScenarioPrediction, ScenarioAnalysis
import sys
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class FactorAttribution:
    """Factor contribution analysis for a prediction"""
    factor_name: str
    contribution_magnitude: float  # How much this factor affects the prediction
    contribution_direction: str    # "positive" or "negative" or "neutral"
    confidence_level: str         # "high", "medium", "low"
    business_impact: str          # Plain English explanation
    

@dataclass
class ExecutiveInsight:
    """Single executive insight with business context"""
    category: str              # "risk", "opportunity", "action", "monitoring"
    priority: str             # "critical", "high", "medium", "low"
    insight: str              # Main insight message
    rationale: str            # Why this insight matters
    recommended_action: str   # What to do about it
    timeframe: str           # "immediate", "short_term", "medium_term"


@dataclass
class ExecutiveExplanation:
    """Complete executive explanation for scenario analysis"""
    executive_summary: str
    key_insights: List[ExecutiveInsight]
    factor_attributions: List[FactorAttribution]
    risk_narrative: str
    opportunity_narrative: str
    recommended_actions: List[str]
    monitoring_points: List[str]
    confidence_statement: str
    

class ExecutiveExplainer:
    """
    Converts technical Bayesian analysis into executive-friendly explanations
    
    Professional Features:
    - Factor attribution analysis using posterior parameter importance
    - Business language translation of statistical concepts  
    - Risk and opportunity narrative generation
    - Actionable recommendations based on predictions
    - Confidence level interpretation for non-technical audiences
    """
    
    def __init__(self, predictor: ScenarioPredictor):
        self.predictor = predictor
        self.feature_names = predictor.feature_names
        
        # Business context mappings
        self.business_context = self._create_business_context()
        self.risk_thresholds = self._define_risk_thresholds()
        
        print("✅ ExecutiveExplainer initialized")
        print(f"🎯 Ready to explain {len(self.feature_names)} economic factors")
        
    def _create_business_context(self) -> Dict:
        """Create business-friendly interpretations of technical factors"""
        return {
            'kospi_return': {
                'name': 'Korean Stock Market Performance',
                'positive_impact': 'broader market optimism lifts 한화솔루션 sentiment',
                'negative_impact': 'market pessimism creates downward pressure',
                'business_relevance': 'Market sentiment affects institutional investor appetite',
                'hedging_strategy': 'Consider market-neutral positions during volatility',
                'monitoring_frequency': 'daily'
            },
            'oil_price_change': {
                'name': 'Oil Price Movement',  
                'positive_impact': 'higher energy costs may benefit renewable energy pivot',
                'negative_impact': 'increased operational costs for chemical operations',
                'business_relevance': 'Direct impact on petrochemical division margins',
                'hedging_strategy': 'Oil price hedging through futures or swaps',
                'monitoring_frequency': 'daily'
            },
            'usd_krw_change': {
                'name': 'USD/KRW Exchange Rate',
                'positive_impact': 'weaker KRW benefits export competitiveness',  
                'negative_impact': 'stronger KRW increases imported raw material costs',
                'business_relevance': 'Currency exposure affects international operations',
                'hedging_strategy': 'Foreign exchange hedging for major transactions',
                'monitoring_frequency': 'daily'
            },
            'vix_change': {
                'name': 'Market Volatility Index',
                'positive_impact': 'lower volatility indicates stable investment environment',
                'negative_impact': 'higher volatility signals market stress and uncertainty',
                'business_relevance': 'Affects cost of capital and investor risk appetite',
                'hedging_strategy': 'Maintain adequate liquidity during high VIX periods', 
                'monitoring_frequency': 'weekly'
            },
            'materials_sector_return': {
                'name': 'Materials Sector Performance',
                'positive_impact': 'sector outperformance indicates strong demand for materials',
                'negative_impact': 'sector underperformance signals weakening industrial demand',
                'business_relevance': 'Direct peer comparison for valuation and performance',
                'hedging_strategy': 'Operational excellence to outperform sector during downturns',
                'monitoring_frequency': 'daily'
            }
        }
    
    def _define_risk_thresholds(self) -> Dict:
        """Define business-relevant risk thresholds"""
        return {
            'return_thresholds': {
                'very_negative': -0.15,  # -15% or worse
                'negative': -0.05,       # -5% to -15%  
                'neutral': 0.05,         # -5% to +5%
                'positive': 0.15,        # +5% to +15%
                'very_positive': 0.15    # +15% or better
            },
            'probability_thresholds': {
                'very_low': 0.2,         # <20% chance
                'low': 0.4,              # 20-40% chance
                'medium': 0.6,           # 40-60% chance  
                'high': 0.8,             # 60-80% chance
                'very_high': 0.8         # >80% chance
            },
            'uncertainty_thresholds': {
                'low': 0.08,             # <8% volatility
                'medium': 0.15,          # 8-15% volatility
                'high': 0.15             # >15% volatility
            }
        }
    
    def analyze_factor_attribution(self, scenario: Scenario, 
                                 prediction: ScenarioPrediction) -> List[FactorAttribution]:
        """
        Analyze which factors drive the prediction through posterior parameter analysis
        """
        attributions = []
        
        # Get posterior parameter estimates from the trained model
        if self.predictor.trace is None:
            return attributions
            
        try:
            # Extract posterior means for each parameter  
            posterior_means = {}
            for i in range(len(self.feature_names) + 1):  # +1 for intercept
                param_name = f'beta_{i}'
                if param_name in self.predictor.trace.posterior.data_vars:
                    samples = self.predictor.trace.posterior[param_name].values.flatten()
                    posterior_means[param_name] = np.mean(samples)
            
            # Convert scenario to feature vector to get input values
            feature_vector = self.predictor.scenario_manager.scenario_to_feature_vector(scenario)
            
            # Calculate contribution of each factor (parameter * input value)
            total_contribution = 0
            factor_contributions = []
            
            for i, feature_name in enumerate(self.feature_names):
                param_key = f'beta_{i+1}'  # +1 because beta_0 is intercept
                if param_key in posterior_means:
                    param_value = posterior_means[param_key]
                    input_value = feature_vector[i]
                    contribution = param_value * input_value
                    
                    factor_contributions.append({
                        'feature': feature_name,
                        'contribution': contribution,
                        'param_value': param_value,
                        'input_value': input_value
                    })
                    total_contribution += abs(contribution)
            
            # Create factor attributions
            for contrib in factor_contributions:
                feature_name = contrib['feature']
                contribution = contrib['contribution'] 
                magnitude = abs(contribution) / max(total_contribution, 1e-6)  # Avoid division by zero
                
                # Determine direction
                if contribution > 0.001:
                    direction = "positive"
                elif contribution < -0.001:
                    direction = "negative"  
                else:
                    direction = "neutral"
                
                # Determine confidence based on magnitude
                if magnitude > 0.3:
                    confidence = "high"
                elif magnitude > 0.1:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                # Get business context
                context = self.business_context.get(feature_name, {})
                if direction == "positive":
                    business_impact = context.get('positive_impact', 'positive impact on returns')
                elif direction == "negative":
                    business_impact = context.get('negative_impact', 'negative impact on returns')
                else:
                    business_impact = 'minimal impact on returns'
                
                attribution = FactorAttribution(
                    factor_name=context.get('name', feature_name),
                    contribution_magnitude=magnitude,
                    contribution_direction=direction,
                    confidence_level=confidence, 
                    business_impact=business_impact
                )
                
                attributions.append(attribution)
                
            # Sort by magnitude (most important first)
            attributions.sort(key=lambda x: x.contribution_magnitude, reverse=True)
                
        except Exception as e:
            print(f"⚠️ Could not compute factor attribution: {e}")
            
        return attributions
    
    def generate_risk_narrative(self, prediction: ScenarioPrediction, 
                               attributions: List[FactorAttribution]) -> str:
        """Generate executive-friendly risk narrative"""
        
        # Risk level interpretation
        if prediction.risk_assessment == "HIGH":
            risk_intro = "This scenario presents significant uncertainty"
        elif prediction.risk_assessment == "MEDIUM":  
            risk_intro = "This scenario shows moderate risk levels"
        else:
            risk_intro = "This scenario appears relatively predictable"
            
        # Uncertainty explanation
        uncertainty_pct = prediction.uncertainty_std * 100
        prob_positive = prediction.probability_positive * 100
        
        uncertainty_context = f"with a ±{uncertainty_pct:.0f}% uncertainty range"
        probability_context = f"There is a {prob_positive:.0f}% chance of positive returns"
        
        # Key risk drivers
        high_impact_factors = [attr for attr in attributions if attr.confidence_level == "high"]
        if high_impact_factors:
            key_driver = high_impact_factors[0]
            driver_context = f"The primary risk driver is {key_driver.factor_name.lower()}, where {key_driver.business_impact}"
        else:
            driver_context = "Risk is distributed across multiple economic factors"
        
        return f"{risk_intro} {uncertainty_context}. {probability_context}. {driver_context}."
    
    def generate_opportunity_narrative(self, prediction: ScenarioPrediction,
                                     attributions: List[FactorAttribution]) -> str:
        """Generate executive-friendly opportunity narrative"""
        
        expected_return_pct = prediction.point_prediction * 100
        # Calculate 90th percentile directly from samples for more reasonable upside case
        upside_90 = np.percentile(prediction.samples, 90) * 100
        
        if expected_return_pct > 5:
            opportunity_intro = f"This scenario offers attractive upside potential with {expected_return_pct:+.1f}% expected returns"
        elif expected_return_pct > 0:
            opportunity_intro = f"This scenario provides modest positive momentum with {expected_return_pct:+.1f}% expected returns"
        else:
            opportunity_intro = f"This scenario presents challenges with {expected_return_pct:+.1f}% expected returns"
            
        upside_context = f"In the best case (90th percentile), returns could reach {upside_90:+.1f}%"
        
        # Identify positive contributors
        positive_factors = [attr for attr in attributions 
                          if attr.contribution_direction == "positive" and attr.confidence_level in ["high", "medium"]]
        
        if positive_factors:
            key_opportunity = positive_factors[0]
            opportunity_driver = f"The primary opportunity driver is {key_opportunity.factor_name.lower()}, where {key_opportunity.business_impact}"
        else:
            opportunity_driver = "Opportunities will depend on broader market conditions"
            
        return f"{opportunity_intro}. {upside_context}. {opportunity_driver}."
    
    def generate_key_insights(self, scenario: Scenario, prediction: ScenarioPrediction,
                            attributions: List[FactorAttribution]) -> List[ExecutiveInsight]:
        """Generate key executive insights"""
        
        insights = []
        
        # Risk insight
        if prediction.risk_assessment == "HIGH":
            risk_insight = ExecutiveInsight(
                category="risk",
                priority="high", 
                insight="High uncertainty requires risk management attention",
                rationale=f"Prediction uncertainty of ±{prediction.uncertainty_std*100:.0f}% indicates volatile conditions",
                recommended_action="Implement risk mitigation strategies and maintain flexibility",
                timeframe="immediate"
            )
            insights.append(risk_insight)
        
        # Probability insight
        if prediction.probability_positive < 0.4:
            probability_insight = ExecutiveInsight(
                category="risk",
                priority="high",
                insight=f"High probability ({(1-prediction.probability_positive)*100:.0f}%) of negative returns", 
                rationale="Market conditions favor downside outcomes in this scenario",
                recommended_action="Consider defensive positioning and downside protection",
                timeframe="short_term"
            )
            insights.append(probability_insight)
        elif prediction.probability_positive > 0.7:
            probability_insight = ExecutiveInsight(
                category="opportunity", 
                priority="medium",
                insight=f"Favorable odds ({prediction.probability_positive*100:.0f}%) for positive returns",
                rationale="Market conditions support upside potential in this scenario", 
                recommended_action="Consider increasing exposure to capture upside",
                timeframe="short_term"
            )
            insights.append(probability_insight)
        
        # Factor-based insights
        for attr in attributions[:2]:  # Top 2 factors
            if attr.confidence_level == "high":
                if attr.contribution_direction == "negative":
                    factor_insight = ExecutiveInsight(
                        category="risk",
                        priority="medium",
                        insight=f"{attr.factor_name} poses significant downside risk",
                        rationale=attr.business_impact.capitalize(),
                        recommended_action=self.business_context.get(
                            attr.factor_name.lower().replace(' ', '_'), {}
                        ).get('hedging_strategy', 'Monitor closely'),
                        timeframe="short_term"
                    )
                    insights.append(factor_insight)
                elif attr.contribution_direction == "positive":
                    factor_insight = ExecutiveInsight(
                        category="opportunity", 
                        priority="medium",
                        insight=f"{attr.factor_name} provides significant upside potential",
                        rationale=attr.business_impact.capitalize(),
                        recommended_action="Position to benefit from this favorable factor",
                        timeframe="short_term"
                    )
                    insights.append(factor_insight)
        
        return insights
    
    def explain_scenario(self, scenario: Scenario, 
                        prediction: ScenarioPrediction) -> ExecutiveExplanation:
        """Create complete executive explanation for a single scenario"""
        
        # Factor attribution analysis
        attributions = self.analyze_factor_attribution(scenario, prediction)
        
        # Generate narratives
        risk_narrative = self.generate_risk_narrative(prediction, attributions)
        opportunity_narrative = self.generate_opportunity_narrative(prediction, attributions)
        
        # Generate insights
        key_insights = self.generate_key_insights(scenario, prediction, attributions)
        
        # Executive summary
        expected_return = prediction.point_prediction * 100
        prob_positive = prediction.probability_positive * 100
        executive_summary = (
            f"Under the '{scenario.name}' scenario, 한화솔루션 is expected to deliver "
            f"{expected_return:+.1f}% returns with {prob_positive:.0f}% probability of gains. "
            f"This scenario carries {prediction.risk_assessment.lower()} risk levels requiring "
            f"{'immediate attention' if prediction.risk_assessment == 'HIGH' else 'standard monitoring'}."
        )
        
        # Recommended actions
        recommended_actions = [insight.recommended_action for insight in key_insights 
                             if insight.priority in ["critical", "high"]]
        
        # Add general recommendations based on risk level
        if prediction.risk_assessment == "HIGH":
            recommended_actions.append("Increase cash reserves and maintain operational flexibility")
        
        if not recommended_actions:
            recommended_actions = ["Continue standard business operations with regular monitoring"]
        
        # Monitoring points  
        monitoring_points = []
        for attr in attributions:
            if attr.confidence_level in ["high", "medium"]:
                context = self.business_context.get(attr.factor_name.lower().replace(' ', '_'), {})
                frequency = context.get('monitoring_frequency', 'weekly')
                monitoring_points.append(f"{attr.factor_name} ({frequency} monitoring)")
        
        # Confidence statement
        high_confidence_factors = sum(1 for attr in attributions if attr.confidence_level == "high")
        total_factors = len(attributions)
        
        if high_confidence_factors >= total_factors * 0.6:
            confidence_statement = "High confidence in prediction drivers and scenario analysis"
        elif high_confidence_factors >= total_factors * 0.3:
            confidence_statement = "Moderate confidence with some uncertainty in key factors"
        else:
            confidence_statement = "Lower confidence due to uncertain factor impacts - monitor closely"
        
        return ExecutiveExplanation(
            executive_summary=executive_summary,
            key_insights=key_insights,
            factor_attributions=attributions,
            risk_narrative=risk_narrative,
            opportunity_narrative=opportunity_narrative,  
            recommended_actions=recommended_actions,
            monitoring_points=monitoring_points,
            confidence_statement=confidence_statement
        )
    
    def create_executive_report(self, scenario: Scenario, prediction: ScenarioPrediction,
                              explanation: ExecutiveExplanation) -> str:
        """Create formatted executive report"""
        
        report = f"""
{'='*80}
🎯 EXECUTIVE SCENARIO ANALYSIS: {scenario.name.upper()}
{'='*80}

📋 EXECUTIVE SUMMARY
{'-'*40}
{explanation.executive_summary}

📊 KEY METRICS  
{'-'*40}
Expected Return:        {prediction.point_prediction:+8.1%}
Probability of Gain:    {prediction.probability_positive:8.1%}
Risk Level:             {prediction.risk_assessment:>8s}
Uncertainty (±1σ):      {prediction.uncertainty_std:8.1%}

90% Confidence Range:   [{prediction.confidence_intervals['5%']:+.1%}, {prediction.confidence_intervals['95%']:+.1%}]

🎯 KEY INSIGHTS
{'-'*40}"""
        
        for i, insight in enumerate(explanation.key_insights, 1):
            priority_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
            category_emoji = {'risk': '⚠️', 'opportunity': '💰', 'action': '🎯', 'monitoring': '👀'}
            
            report += f"""
{i}. {category_emoji.get(insight.category, '📌')} {priority_emoji.get(insight.priority, '⚪')} {insight.insight}
   Rationale: {insight.rationale}
   Action: {insight.recommended_action}
   Timeframe: {insight.timeframe.replace('_', ' ').title()}"""

        report += f"""

📈 FACTOR ATTRIBUTION ANALYSIS
{'-'*40}"""
        
        for attr in explanation.factor_attributions:
            direction_emoji = {'positive': '📈', 'negative': '📉', 'neutral': '➡️'}
            confidence_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            
            report += f"""
{direction_emoji[attr.contribution_direction]} {attr.factor_name} ({attr.contribution_magnitude:.1%} impact)
   {confidence_emoji[attr.confidence_level]} Confidence: {attr.confidence_level.upper()}
   Business Impact: {attr.business_impact.capitalize()}"""
        
        report += f"""

⚠️  RISK ASSESSMENT
{'-'*40}
{explanation.risk_narrative}

💰 OPPORTUNITY ASSESSMENT  
{'-'*40}
{explanation.opportunity_narrative}

🎯 RECOMMENDED ACTIONS
{'-'*40}"""
        
        for i, action in enumerate(explanation.recommended_actions, 1):
            report += f"""
{i}. {action}"""

        report += f"""

👀 KEY MONITORING POINTS
{'-'*40}"""
        
        for point in explanation.monitoring_points:
            report += f"""
• {point}"""
        
        report += f"""

🎲 CONFIDENCE ASSESSMENT
{'-'*40}
{explanation.confidence_statement}

{'='*80}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        
        return report
    
    def display_executive_explanation(self, scenario: Scenario, prediction: ScenarioPrediction):
        """Display executive explanation interactively"""
        explanation = self.explain_scenario(scenario, prediction)
        report = self.create_executive_report(scenario, prediction, explanation)
        print(report)
        return explanation


def demo_executive_explainer():
    """Demonstrate executive explanation capabilities"""
    print("💼 EXECUTIVE EXPLANATION ENGINE DEMONSTRATION")
    print("="*70)
    
    # Initialize predictor and explainer
    predictor = ScenarioPredictor(model_type="elicited") 
    explainer = ExecutiveExplainer(predictor)
    
    # Get a scenario and prediction
    scenario = predictor.scenario_manager.get_template('bull_market')
    prediction = predictor.predict_scenario(scenario, n_samples=500)
    
    # Generate and display executive explanation
    print(f"\n🎯 GENERATING EXECUTIVE EXPLANATION FOR: {scenario.name}")
    explanation = explainer.display_executive_explanation(scenario, prediction)
    
    print(f"\n✅ Stage 3 demonstration complete!")
    print(f"Next: Stage 4 - Decision Support Dashboard")


if __name__ == "__main__":
    demo_executive_explainer() 
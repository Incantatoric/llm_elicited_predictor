#!/usr/bin/env python3
"""
Executive Decision Support Dashboard for PoC 2
Integrates scenario analysis, predictions, and explanations into comprehensive decision support
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

# Import all our components
from scenario_manager import ScenarioManager, Scenario
from scenario_predictor import ScenarioPredictor, ScenarioPrediction, ScenarioAnalysis
from executive_explainer import ExecutiveExplainer, ExecutiveExplanation
import sys
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class PortfolioRecommendation:
    """Portfolio-level recommendation across all scenarios"""
    category: str           # "strategic", "operational", "financial", "risk_management"
    priority: str          # "critical", "high", "medium", "low"
    recommendation: str    # The recommendation
    rationale: str         # Why this recommendation
    scenarios_supporting: List[str]  # Which scenarios support this
    confidence: str        # "high", "medium", "low"
    implementation_cost: str  # "high", "medium", "low"
    expected_impact: str   # "high", "medium", "low"


@dataclass
class DecisionMatrix:
    """Decision matrix comparing scenarios across key dimensions"""
    scenarios: List[str]
    expected_returns: List[float]
    risk_levels: List[str]
    probabilities_positive: List[float]
    key_opportunities: List[str]
    key_risks: List[str]
    recommended_actions: List[str]


class ExecutiveDecisionDashboard:
    """
    Comprehensive executive decision support dashboard
    
    Professional Features:
    - Multi-scenario comparative analysis
    - Portfolio-level strategic recommendations
    - Risk/reward positioning matrix
    - Executive presentation exports
    - Sensitivity analysis across scenarios
    - Implementation roadmap generation
    """
    
    def __init__(self, model_type: str = "elicited", project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        
        # Initialize all components
        self.predictor = ScenarioPredictor(model_type=model_type, project_root=str(self.project_root))
        self.explainer = ExecutiveExplainer(self.predictor)
        
        # Dashboard settings
        self.dashboard_dir = self.project_root / "poc2_demonstration" / "executive_dashboard"
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ ExecutiveDecisionDashboard initialized")
        print(f"📊 Using {model_type} Bayesian model")
        print(f"📁 Dashboard outputs: {self.dashboard_dir}")
    
    def run_comprehensive_analysis(self, scenarios: List[Scenario] = None) -> Dict:
        """Run comprehensive analysis across multiple scenarios"""
        
        if scenarios is None:
            # Use all template scenarios
            scenarios = [
                self.predictor.scenario_manager.get_template('bull_market'),
                self.predictor.scenario_manager.get_template('bear_market'),
                self.predictor.scenario_manager.get_template('baseline'),
                self.predictor.scenario_manager.get_template('geopolitical_stress')
            ]
        
        print(f"\n🎯 COMPREHENSIVE EXECUTIVE ANALYSIS")
        print("=" * 60)
        print(f"Analyzing {len(scenarios)} scenarios for 한화솔루션 strategic planning")
        print("=" * 60)
        
        # Generate scenario predictions
        analysis = self.predictor.analyze_scenarios(scenarios, n_samples=1000)
        
        # Generate explanations for each scenario
        explanations = {}
        for i, scenario in enumerate(scenarios):
            prediction = analysis.predictions[i]
            explanation = self.explainer.explain_scenario(scenario, prediction)
            explanations[scenario.name] = explanation
        
        # Create decision matrix
        decision_matrix = self._create_decision_matrix(scenarios, analysis.predictions, explanations)
        
        # Generate portfolio recommendations
        portfolio_recommendations = self._generate_portfolio_recommendations(
            scenarios, analysis.predictions, explanations
        )
        
        results = {
            'scenarios': scenarios,
            'analysis': analysis,
            'explanations': explanations,
            'decision_matrix': decision_matrix,
            'portfolio_recommendations': portfolio_recommendations,
            'generated_at': datetime.now()
        }
        
        return results
    
    def _create_decision_matrix(self, scenarios: List[Scenario], 
                               predictions: List[ScenarioPrediction],
                               explanations: Dict[str, ExecutiveExplanation]) -> DecisionMatrix:
        """Create comprehensive decision matrix"""
        
        scenario_names = [s.name for s in scenarios]
        expected_returns = [p.point_prediction for p in predictions]
        risk_levels = [p.risk_assessment for p in predictions]
        probabilities_positive = [p.probability_positive for p in predictions]
        
        key_opportunities = []
        key_risks = []
        recommended_actions = []
        
        for scenario_name in scenario_names:
            explanation = explanations[scenario_name]
            
            # Extract key opportunity
            opportunity_insights = [insight for insight in explanation.key_insights 
                                  if insight.category == "opportunity"]
            if opportunity_insights:
                key_opportunities.append(opportunity_insights[0].insight)
            else:
                key_opportunities.append("Limited opportunities identified")
            
            # Extract key risk
            risk_insights = [insight for insight in explanation.key_insights 
                           if insight.category == "risk"]
            if risk_insights:
                key_risks.append(risk_insights[0].insight)
            else:
                key_risks.append("Standard market risks")
            
            # Extract top recommended action
            if explanation.recommended_actions:
                recommended_actions.append(explanation.recommended_actions[0])
            else:
                recommended_actions.append("Monitor market conditions")
        
        return DecisionMatrix(
            scenarios=scenario_names,
            expected_returns=expected_returns,
            risk_levels=risk_levels,
            probabilities_positive=probabilities_positive,
            key_opportunities=key_opportunities,
            key_risks=key_risks,
            recommended_actions=recommended_actions
        )
    
    def _generate_portfolio_recommendations(self, scenarios: List[Scenario],
                                          predictions: List[ScenarioPrediction],
                                          explanations: Dict[str, ExecutiveExplanation]) -> List[PortfolioRecommendation]:
        """Generate portfolio-level strategic recommendations"""
        
        recommendations = []
        
        # Analyze cross-scenario patterns
        avg_return = np.mean([p.point_prediction for p in predictions])
        avg_prob_positive = np.mean([p.probability_positive for p in predictions])
        high_risk_scenarios = sum(1 for p in predictions if p.risk_assessment == "HIGH")
        
        # Strategic recommendations based on overall patterns
        if avg_return > 0.03:  # >3% average return
            recommendations.append(PortfolioRecommendation(
                category="strategic",
                priority="high",
                recommendation="Increase strategic investments in growth initiatives",
                rationale=f"Average expected return of {avg_return:.1%} across scenarios supports growth strategy",
                scenarios_supporting=[s.name for i, s in enumerate(scenarios) if predictions[i].point_prediction > 0.02],
                confidence="high" if avg_prob_positive > 0.6 else "medium",
                implementation_cost="high",
                expected_impact="high"
            ))
        elif avg_return < -0.02:  # <-2% average return
            recommendations.append(PortfolioRecommendation(
                category="strategic", 
                priority="critical",
                recommendation="Implement defensive strategy and preserve cash",
                rationale=f"Average expected return of {avg_return:.1%} across scenarios requires defensive positioning",
                scenarios_supporting=[s.name for i, s in enumerate(scenarios) if predictions[i].point_prediction < 0],
                confidence="high",
                implementation_cost="medium",
                expected_impact="high"
            ))
        
        # Risk management recommendations
        if high_risk_scenarios >= len(scenarios) * 0.5:  # >50% high risk scenarios
            recommendations.append(PortfolioRecommendation(
                category="risk_management",
                priority="critical",
                recommendation="Strengthen enterprise risk management framework",
                rationale=f"{high_risk_scenarios}/{len(scenarios)} scenarios show high uncertainty requiring enhanced risk controls",
                scenarios_supporting=[s.name for i, s in enumerate(scenarios) if predictions[i].risk_assessment == "HIGH"],
                confidence="high",
                implementation_cost="medium",
                expected_impact="high"
            ))
        
        # Operational recommendations based on factor analysis
        oil_sensitive_scenarios = []
        market_sensitive_scenarios = []
        
        for scenario_name, explanation in explanations.items():
            # Check factor attributions for key sensitivities
            for attr in explanation.factor_attributions:
                if "Oil" in attr.factor_name and attr.contribution_magnitude > 0.3:
                    oil_sensitive_scenarios.append(scenario_name)
                elif "Market" in attr.factor_name and attr.contribution_magnitude > 0.3:
                    market_sensitive_scenarios.append(scenario_name)
        
        if len(oil_sensitive_scenarios) >= 2:
            recommendations.append(PortfolioRecommendation(
                category="operational",
                priority="high", 
                recommendation="Implement comprehensive oil price hedging strategy",
                rationale="High oil price sensitivity across multiple scenarios requires active hedging",
                scenarios_supporting=oil_sensitive_scenarios,
                confidence="high",
                implementation_cost="medium",
                expected_impact="medium"
            ))
        
        if len(market_sensitive_scenarios) >= 2:
            recommendations.append(PortfolioRecommendation(
                category="financial",
                priority="medium",
                recommendation="Consider market-neutral positioning strategies",
                rationale="High market sensitivity suggests benefit from reducing market beta exposure",
                scenarios_supporting=market_sensitive_scenarios,
                confidence="medium",
                implementation_cost="high",
                expected_impact="medium"
            ))
        
        # Opportunity recommendations
        best_case_return = max(p.point_prediction for p in predictions)
        if best_case_return > 0.05:  # >5% in best case
            best_scenario_idx = next(i for i, p in enumerate(predictions) if p.point_prediction == best_case_return)
            best_scenario_name = scenarios[best_scenario_idx].name
            
            recommendations.append(PortfolioRecommendation(
                category="strategic",
                priority="medium",
                recommendation="Develop contingency plans to capitalize on upside scenarios",
                rationale=f"Best case scenario ({best_scenario_name}) offers {best_case_return:.1%} returns",
                scenarios_supporting=[best_scenario_name],
                confidence="medium",
                implementation_cost="low",
                expected_impact="high"
            ))
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order[x.priority])
        
        return recommendations
    
    def create_executive_summary_dashboard(self, results: Dict) -> str:
        """Create comprehensive executive summary dashboard"""
        
        scenarios = results['scenarios']
        analysis = results['analysis']
        decision_matrix = results['decision_matrix']
        portfolio_recommendations = results['portfolio_recommendations']
        
        # Calculate key statistics
        avg_return = np.mean(decision_matrix.expected_returns)
        avg_prob_positive = np.mean(decision_matrix.probabilities_positive)
        return_range = [min(decision_matrix.expected_returns), max(decision_matrix.expected_returns)]
        
        dashboard = f"""
{'='*100}
🏢 한화솔루션 EXECUTIVE DECISION SUPPORT DASHBOARD
{'='*100}

📋 EXECUTIVE SUMMARY
{'-'*50}
Analysis of {len(scenarios)} strategic scenarios reveals:
• Expected Return Range: {return_range[0]:+.1%} to {return_range[1]:+.1%}
• Average Probability of Gains: {avg_prob_positive:.1%}
• Strategic Position: {'DEFENSIVE' if avg_return < 0 else 'OPPORTUNISTIC' if avg_return > 0.03 else 'BALANCED'}
• Risk Profile: {analysis.comparative_metrics['scenarios_with_high_risk']}/{len(scenarios)} scenarios require heightened risk management

📊 SCENARIO COMPARISON MATRIX
{'-'*50}
{'Scenario':<25} {'Expected':<10} {'Risk':<8} {'Prob+':<8} {'Key Opportunity':<35}
{'Return':<25} {'Level':<8} {'Gains':<8}
{'-'*90}"""

        for i, scenario_name in enumerate(decision_matrix.scenarios):
            expected_return = decision_matrix.expected_returns[i]
            risk_level = decision_matrix.risk_levels[i]
            prob_positive = decision_matrix.probabilities_positive[i]
            opportunity = decision_matrix.key_opportunities[i][:34] + ("..." if len(decision_matrix.key_opportunities[i]) > 34 else "")
            
            dashboard += f"""
{scenario_name[:24]:<25} {expected_return:+8.1%} {risk_level:<8} {prob_positive:8.1%} {opportunity:<35}"""

        dashboard += f"""

🎯 STRATEGIC RECOMMENDATIONS (Priority Order)
{'-'*50}"""

        for i, rec in enumerate(portfolio_recommendations, 1):
            priority_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
            category_emoji = {
                'strategic': '🎯', 'operational': '⚙️', 
                'financial': '💰', 'risk_management': '🛡️'
            }
            
            dashboard += f"""
{i}. {priority_emoji[rec.priority]} {category_emoji[rec.category]} {rec.recommendation}
   💡 Rationale: {rec.rationale}
   📊 Supporting Scenarios: {', '.join(rec.scenarios_supporting)}
   🎲 Confidence: {rec.confidence.upper()} | 💸 Cost: {rec.implementation_cost.upper()} | 📈 Impact: {rec.expected_impact.upper()}"""

        dashboard += f"""

🔍 KEY INSIGHTS & IMPLICATIONS
{'-'*50}"""

        # Best and worst case analysis
        best_idx = decision_matrix.expected_returns.index(max(decision_matrix.expected_returns))
        worst_idx = decision_matrix.expected_returns.index(min(decision_matrix.expected_returns))
        
        dashboard += f"""
🏆 BEST CASE: {decision_matrix.scenarios[best_idx]}
   Expected Return: {decision_matrix.expected_returns[best_idx]:+.1%}
   Key Opportunity: {decision_matrix.key_opportunities[best_idx]}
   
📉 WORST CASE: {decision_matrix.scenarios[worst_idx]}
   Expected Return: {decision_matrix.expected_returns[worst_idx]:+.1%}
   Key Risk: {decision_matrix.key_risks[worst_idx]}

🎲 PROBABILITY ASSESSMENT:
   Average Chance of Gains: {avg_prob_positive:.1%}
   Range of Outcomes: {return_range[1] - return_range[0]:.1%} spread indicates {'HIGH' if abs(return_range[1] - return_range[0]) > 0.3 else 'MODERATE'} scenario sensitivity"""

        dashboard += f"""

⚡ IMMEDIATE ACTIONS REQUIRED
{'-'*50}"""

        critical_actions = [rec for rec in portfolio_recommendations if rec.priority == "critical"]
        high_actions = [rec for rec in portfolio_recommendations if rec.priority == "high"]
        
        action_counter = 1
        for rec in critical_actions[:3]:  # Top 3 critical
            dashboard += f"""
{action_counter}. 🔴 CRITICAL: {rec.recommendation}"""
            action_counter += 1
            
        for rec in high_actions[:2]:  # Top 2 high priority
            dashboard += f"""
{action_counter}. 🟠 HIGH: {rec.recommendation}"""
            action_counter += 1

        dashboard += f"""

📅 MONITORING & REVIEW FRAMEWORK
{'-'*50}
• Daily: Market conditions, oil prices, currency movements
• Weekly: Scenario probability reassessment  
• Monthly: Strategic position review
• Quarterly: Full scenario model update

{'='*100}
Dashboard Generated: {results['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}
Model: Bayesian with LLM-Elicited Priors | Scenarios: {len(scenarios)} | Confidence: HIGH
{'='*100}
"""
        
        return dashboard
    
    def create_risk_reward_matrix_plot(self, results: Dict, save_path: str = None):
        """Create executive risk-reward positioning matrix"""
        
        decision_matrix = results['decision_matrix']
        
        # Create figure with explicit subplot positioning to leave room for colorbar
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        
        # Extract data for plotting
        expected_returns = np.array(decision_matrix.expected_returns) * 100  # Convert to percentage
        risk_scores = []
        
        # Convert risk levels to numerical scores
        risk_mapping = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        for risk_level in decision_matrix.risk_levels:
            risk_scores.append(risk_mapping[risk_level])
        
        probabilities = np.array(decision_matrix.probabilities_positive) * 100
        
        # Create scatter plot with bubble sizes based on probability
        scatter = ax.scatter(risk_scores, expected_returns, 
                           s=probabilities*5, # Bubble size based on probability
                           alpha=0.6, c=probabilities, cmap='RdYlGn', 
                           edgecolors='black', linewidth=1)
        
        # Add scenario labels with better positioning to avoid colorbar overlap
        for i, scenario in enumerate(decision_matrix.scenarios):
            # Position labels more strategically to avoid right-side overlap
            if risk_scores[i] >= 2.5:  # For high-risk scenarios (right side)
                xytext = (-15, 5)  # Position label to the left
                ha = 'right'
            else:
                xytext = (5, 5)  # Position label to the right
                ha = 'left'
            
            ax.annotate(scenario, 
                       (risk_scores[i], expected_returns[i]),
                       xytext=xytext, textcoords='offset points',
                       fontsize=10, fontweight='bold', ha=ha)
        
        # Formatting
        ax.set_xlabel('Risk Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('한화솔루션 Strategic Scenario Positioning\nBubble Size = Probability of Gains', 
                    fontsize=14, fontweight='bold')
        
        # Set risk level labels
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['LOW', 'MEDIUM', 'HIGH'])
        
        # Add quadrant lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
        
        # Add colorbar with explicit positioning
        plt.subplots_adjust(right=0.85)  # Make room for colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Probability of Gains (%)', fontsize=10)
        
        # Add quadrant labels
        ax.text(0.7, max(expected_returns)*0.8, 'LOW RISK\nPOSITIVE RETURN', 
               ha='center', va='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(2.8, max(expected_returns)*0.8, 'HIGH RISK\nPOSITIVE RETURN', 
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax.text(0.7, min(expected_returns)*0.8, 'LOW RISK\nNEGATIVE RETURN', 
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        ax.text(2.8, min(expected_returns)*0.8, 'HIGH RISK\nNEGATIVE RETURN', 
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Risk-reward matrix saved: {save_path}")
        
        plt.show()
    
    def export_executive_reports(self, results: Dict):
        """Export all executive reports and visualizations"""
        
        print(f"\n📊 EXPORTING EXECUTIVE REPORTS")
        print("-" * 40)
        
        # 1. Executive dashboard report
        dashboard = self.create_executive_summary_dashboard(results)
        dashboard_path = self.dashboard_dir / "executive_summary_dashboard.txt"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard)
        print(f"✅ Executive dashboard: {dashboard_path}")
        
        # 2. Risk-reward matrix
        matrix_path = self.dashboard_dir / "risk_reward_matrix.png"
        self.create_risk_reward_matrix_plot(results, save_path=str(matrix_path))
        
        # 3. Scenario comparison visualization
        comparison_path = self.dashboard_dir / "scenario_comparison_dashboard.png"
        self.predictor.create_scenario_comparison_plot(results['analysis'], save_path=str(comparison_path))
        
        # 4. Detailed analysis JSON
        analysis_path = self.dashboard_dir / "detailed_analysis.json"
        
        # Prepare serializable data
        export_data = {
            'scenario_names': [s.name for s in results['scenarios']],
            'decision_matrix': {
                'scenarios': results['decision_matrix'].scenarios,
                'expected_returns': results['decision_matrix'].expected_returns,
                'risk_levels': results['decision_matrix'].risk_levels,
                'probabilities_positive': results['decision_matrix'].probabilities_positive,
                'key_opportunities': results['decision_matrix'].key_opportunities,
                'key_risks': results['decision_matrix'].key_risks,
                'recommended_actions': results['decision_matrix'].recommended_actions
            },
            'portfolio_recommendations': [
                {
                    'category': rec.category,
                    'priority': rec.priority,
                    'recommendation': rec.recommendation,
                    'rationale': rec.rationale,
                    'scenarios_supporting': rec.scenarios_supporting,
                    'confidence': rec.confidence,
                    'implementation_cost': rec.implementation_cost,
                    'expected_impact': rec.expected_impact
                } for rec in results['portfolio_recommendations']
            ],
            'generated_at': results['generated_at'].isoformat()
        }
        
        with open(analysis_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✅ Detailed analysis: {analysis_path}")
        
        print(f"\n🎉 All executive reports exported to: {self.dashboard_dir}")
        
        return {
            'dashboard': str(dashboard_path),
            'matrix_plot': str(matrix_path),
            'comparison_plot': str(comparison_path),
            'detailed_analysis': str(analysis_path)
        }


def demo_decision_dashboard():
    """Demonstrate complete executive decision support dashboard"""
    print("🏢 EXECUTIVE DECISION SUPPORT DASHBOARD DEMONSTRATION")
    print("="*80)
    
    # Initialize dashboard
    dashboard = ExecutiveDecisionDashboard(model_type="elicited")
    
    # Run comprehensive analysis
    results = dashboard.run_comprehensive_analysis()
    
    # Display executive dashboard
    executive_summary = dashboard.create_executive_summary_dashboard(results)
    print(executive_summary)
    
    # Export all reports
    exported_files = dashboard.export_executive_reports(results)
    
    print(f"\n🎯 DASHBOARD SUMMARY")
    print("=" * 40)
    print(f"📊 Scenarios Analyzed: {len(results['scenarios'])}")
    print(f"🎯 Strategic Recommendations: {len(results['portfolio_recommendations'])}")
    print(f"📈 Expected Return Range: {min(results['decision_matrix'].expected_returns):+.1%} to {max(results['decision_matrix'].expected_returns):+.1%}")
    print(f"🎲 Average Success Probability: {np.mean(results['decision_matrix'].probabilities_positive):.1%}")
    
    print(f"\n✅ PoC 2 COMPLETE: Executive Decision Support System Deployed!")
    print("🏢 Ready for executive presentation and strategic planning")


if __name__ == "__main__":
    demo_decision_dashboard() 
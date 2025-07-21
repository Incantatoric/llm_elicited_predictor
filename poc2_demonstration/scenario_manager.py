#!/usr/bin/env python3
"""
Scenario Manager for PoC 2: Executive Decision Support
Handles scenario definition, validation, and management for what-if analysis
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings


@dataclass
class ScenarioCondition:
    """Individual condition within a scenario"""
    variable: str
    value: float
    description: str
    confidence: str = "medium"  # low, medium, high


@dataclass  
class Scenario:
    """Complete scenario definition"""
    name: str
    description: str
    conditions: List[ScenarioCondition]
    probability: Optional[float] = None
    time_horizon: str = "1_quarter"
    created_by: str = "analyst"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ScenarioManager:
    """
    Manages scenario creation, validation, and storage for executive analysis
    
    Professional Features:
    - Realistic range validation
    - Pre-built scenario templates  
    - Historical data-informed constraints
    - Batch scenario operations
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.scenarios_dir = self.project_root / "poc2_demonstration" / "scenarios"
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        
        # Load feature metadata for validation
        self.feature_info = self._load_feature_metadata()
        self.historical_ranges = self._calculate_historical_ranges()
        
        # Predefined scenario templates
        self.templates = self._create_scenario_templates()
        
        print(f"✅ ScenarioManager initialized")
        print(f"📁 Scenarios directory: {self.scenarios_dir}")
        print(f"📊 Available features: {list(self.feature_info.keys())}")
    
    def _load_feature_metadata(self) -> Dict:
        """Load feature information from processed data"""
        try:
            with open(self.project_root / 'data/processed/metadata.json') as f:
                metadata = json.load(f)
            return {name: {'index': i} for i, name in enumerate(metadata['feature_names'])}
        except Exception as e:
            warnings.warn(f"Could not load feature metadata: {e}")
            # Fallback feature list based on our known structure
            return {
                'kospi_return': {'index': 0},
                'oil_price_change': {'index': 1}, 
                'usd_krw_change': {'index': 2},
                'vix_change': {'index': 3},
                'materials_sector_return': {'index': 4}
            }
    
    def _calculate_historical_ranges(self) -> Dict:
        """Calculate realistic ranges based on historical data"""
        try:
            # Load historical data to understand realistic ranges
            X = pd.read_csv(self.project_root / 'data/processed/features_standardized.csv', index_col=0)
            
            ranges = {}
            for col in X.columns:
                ranges[col] = {
                    'min': float(X[col].min()),
                    'max': float(X[col].max()),
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std()),
                    'p05': float(X[col].quantile(0.05)),
                    'p95': float(X[col].quantile(0.95)),
                    'realistic_min': float(X[col].quantile(0.01)),  # 1st percentile
                    'realistic_max': float(X[col].quantile(0.99))   # 99th percentile
                }
            return ranges
            
        except Exception as e:
            warnings.warn(f"Could not calculate historical ranges: {e}")
            return {}
    
    def _create_scenario_templates(self) -> Dict:
        """Create professional scenario templates"""
        templates = {}
        
        # Bull Market Scenario
        templates['bull_market'] = Scenario(
            name="Bull Market Recovery",
            description="Optimistic scenario with economic recovery and market growth",
            conditions=[
                ScenarioCondition('kospi_return', 0.15, 'KOSPI rises 15% due to economic recovery', 'high'),
                ScenarioCondition('oil_price_change', -0.15, 'Oil prices drop 15% on demand destruction', 'medium'),
                ScenarioCondition('usd_krw_change', -0.05, 'KRW strengthens 5% vs USD', 'medium'),
                ScenarioCondition('vix_change', -0.20, 'VIX decreases 20% as volatility subsides', 'high'),
                ScenarioCondition('materials_sector_return', 0.12, 'Materials sector outperforms 12%', 'medium')
            ],
            probability=0.25,
            tags=['optimistic', 'recovery', 'growth']
        )
        
        # Bear Market Scenario  
        templates['bear_market'] = Scenario(
            name="Market Correction",
            description="Pessimistic scenario with economic downturn and market stress",
            conditions=[
                ScenarioCondition('kospi_return', -0.20, 'KOSPI falls 20% on economic concerns', 'medium'),
                ScenarioCondition('oil_price_change', -0.10, 'Oil prices increase 10% on demand recovery', 'medium'),
                ScenarioCondition('usd_krw_change', 0.08, 'KRW weakens 8% vs USD on capital flight', 'high'),
                ScenarioCondition('vix_change', 0.40, 'VIX spikes 40% on market fear', 'high'),
                ScenarioCondition('materials_sector_return', -0.18, 'Materials sector underperforms -18%', 'high')
            ],
            probability=0.20,
            tags=['pessimistic', 'correction', 'stress']
        )
        
        # Baseline Scenario
        templates['baseline'] = Scenario(
            name="Baseline Continuation", 
            description="Expected scenario based on current trends continuation",
            conditions=[
                ScenarioCondition('kospi_return', 0.02, 'KOSPI modest growth 2%', 'high'),
                ScenarioCondition('oil_price_change', 0.01, 'Oil prices stable with 1% increase', 'medium'),
                ScenarioCondition('usd_krw_change', 0.01, 'USD/KRW stable with 1% depreciation', 'medium'),
                ScenarioCondition('vix_change', 0.05, 'VIX slightly elevated 5%', 'medium'),
                ScenarioCondition('materials_sector_return', 0.03, 'Materials sector in-line 3%', 'medium')
            ],
            probability=0.40,
            tags=['baseline', 'continuation', 'expected']
        )
        
        # Geopolitical Stress  
        templates['geopolitical_stress'] = Scenario(
            name="Geopolitical Tensions",
            description="Scenario with heightened geopolitical risks affecting markets",
            conditions=[
                ScenarioCondition('kospi_return', -0.10, 'KOSPI down 10% on regional tensions', 'medium'),
                ScenarioCondition('oil_price_change', 0.25, 'Oil prices spike 25% on supply concerns', 'high'),
                ScenarioCondition('usd_krw_change', 0.06, 'KRW weakens 6% on safe-haven flows', 'high'),
                ScenarioCondition('vix_change', 0.35, 'VIX jumps 35% on uncertainty', 'high'),
                ScenarioCondition('materials_sector_return', -0.05, 'Materials modestly down 5%', 'medium')
            ],
            probability=0.15,
            tags=['geopolitical', 'stress', 'uncertainty']
        )
        
        return templates
    
    def validate_scenario(self, scenario: Scenario) -> Tuple[bool, List[str]]:
        """Validate scenario conditions against historical ranges"""
        issues = []
        
        for condition in scenario.conditions:
            var_name = condition.variable
            
            # Check if variable exists
            if var_name not in self.feature_info:
                issues.append(f"Unknown variable: {var_name}")
                continue
                
            # Check against historical ranges if available
            if var_name in self.historical_ranges:
                ranges = self.historical_ranges[var_name]
                value = condition.value
                
                # Check if extremely unrealistic (beyond 99th percentile range)
                if value < ranges['realistic_min'] or value > ranges['realistic_max']:
                    issues.append(
                        f"{var_name}={value:.3f} is outside realistic range "
                        f"[{ranges['realistic_min']:.3f}, {ranges['realistic_max']:.3f}]"
                    )
                
                # Check if moderately unrealistic (beyond 95th percentile)  
                elif value < ranges['p05'] or value > ranges['p95']:
                    issues.append(
                        f"{var_name}={value:.3f} is in extreme range "
                        f"(historical 95%: [{ranges['p05']:.3f}, {ranges['p95']:.3f}])"
                    )
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def create_custom_scenario(self, name: str, description: str, 
                             conditions: Dict[str, float]) -> Scenario:
        """Create a custom scenario from variable-value pairs"""
        scenario_conditions = []
        
        for var_name, value in conditions.items():
            if var_name not in self.feature_info:
                raise ValueError(f"Unknown variable: {var_name}")
                
            scenario_conditions.append(
                ScenarioCondition(
                    variable=var_name,
                    value=value,
                    description=f"{var_name} set to {value:.3f}"
                )
            )
        
        scenario = Scenario(
            name=name,
            description=description,
            conditions=scenario_conditions,
            created_by="custom"
        )
        
        # Validate the scenario
        is_valid, issues = self.validate_scenario(scenario)
        if not is_valid:
            print("⚠️ Scenario validation warnings:")
            for issue in issues:
                print(f"  • {issue}")
        
        return scenario
    
    def get_template(self, template_name: str) -> Scenario:
        """Get a predefined scenario template"""
        if template_name not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Unknown template: {template_name}. Available: {available}")
        
        return self.templates[template_name]
    
    def list_templates(self) -> Dict[str, str]:
        """List available scenario templates"""
        return {name: scenario.description for name, scenario in self.templates.items()}
    
    def scenario_to_feature_vector(self, scenario: Scenario) -> np.ndarray:
        """Convert scenario to feature vector for model input"""
        # Create array with all zeros (standardized data has mean ~0)
        feature_vector = np.zeros(len(self.feature_info))
        
        # Set values from scenario conditions
        for condition in scenario.conditions:
            if condition.variable in self.feature_info:
                idx = self.feature_info[condition.variable]['index']
                feature_vector[idx] = condition.value
            
        return feature_vector
    
    def save_scenario(self, scenario: Scenario) -> str:
        """Save scenario to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scenario_{scenario.name.replace(' ', '_').lower()}_{timestamp}.json"
        filepath = self.scenarios_dir / filename
        
        # Convert to serializable format
        scenario_dict = {
            'name': scenario.name,
            'description': scenario.description,
            'conditions': [
                {
                    'variable': c.variable,
                    'value': c.value,
                    'description': c.description,
                    'confidence': c.confidence
                } for c in scenario.conditions
            ],
            'probability': scenario.probability,
            'time_horizon': scenario.time_horizon,
            'created_by': scenario.created_by,
            'tags': scenario.tags,
            'created_at': timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(scenario_dict, f, indent=2)
        
        print(f"✅ Scenario saved: {filepath}")
        return str(filepath)
    
    def display_scenario_summary(self, scenario: Scenario):
        """Display executive-friendly scenario summary"""
        print(f"\n{'='*60}")
        print(f"📋 SCENARIO: {scenario.name}")
        print(f"{'='*60}")
        print(f"📝 Description: {scenario.description}")
        
        if scenario.probability:
            print(f"🎯 Probability: {scenario.probability:.1%}")
            
        if scenario.tags:
            print(f"🏷️  Tags: {', '.join(scenario.tags)}")
        
        print(f"\n📊 CONDITIONS:")
        print("-" * 40)
        
        for condition in scenario.conditions:
            confidence_emoji = {'low': '🟡', 'medium': '🟠', 'high': '🔴'}
            emoji = confidence_emoji.get(condition.confidence, '⚪')
            
            print(f"{emoji} {condition.variable:<25} {condition.value:>8.3f}")
            print(f"   {condition.description}")
        
        print("-" * 40)
        
        # Show validation status
        is_valid, issues = self.validate_scenario(scenario)
        if is_valid:
            print("✅ Scenario validation: PASSED")
        else:
            print("⚠️ Scenario validation: WARNINGS")
            for issue in issues:
                print(f"   • {issue}")


def demo_scenario_manager():
    """Demonstration of scenario manager capabilities"""
    print("🎯 SCENARIO MANAGER DEMONSTRATION")
    print("="*50)
    
    # Initialize manager
    manager = ScenarioManager()
    
    # 1. Show available templates
    print(f"\n📋 AVAILABLE TEMPLATES:")
    templates = manager.list_templates()
    for name, desc in templates.items():
        print(f"  • {name}: {desc}")
    
    # 2. Display a template scenario
    bull_scenario = manager.get_template('bull_market')
    manager.display_scenario_summary(bull_scenario)
    
    # 3. Create custom scenario
    print(f"\n🛠️ CREATING CUSTOM SCENARIO")
    custom_conditions = {
        'kospi_return': 0.08,
        'oil_price_change': 0.20, 
        'vix_change': 0.15
    }
    
    custom_scenario = manager.create_custom_scenario(
        name="Oil Price Shock",
        description="Scenario with significant oil price increase",
        conditions=custom_conditions
    )
    
    manager.display_scenario_summary(custom_scenario)
    
    # 4. Convert to feature vector (ready for model input)
    feature_vector = manager.scenario_to_feature_vector(custom_scenario)
    print(f"\n🔢 FEATURE VECTOR FOR MODEL:")
    print(f"Shape: {feature_vector.shape}")
    print(f"Values: {feature_vector}")
    
    print(f"\n✅ Stage 1 demonstration complete!")
    print(f"Next: Stage 2 - Scenario-Based Prediction Engine")


if __name__ == "__main__":
    demo_scenario_manager() 
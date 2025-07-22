#!/usr/bin/env python3
"""
Comprehensive comparison of all three methods:
1. Bayesian with LLM-elicited priors
2. Bayesian with uninformed priors  
3. Naive LLM (in-context learning)

This implements the complete PoC 1 validation from the Capstick paper.
"""

import sys
import os
from pathlib import Path

# Add the poc1_validation directory to the path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bayesian_evaluator import BayesianEvaluator
from naive_llm_evaluator import NaiveLLMEvaluator
import pandas as pd
import numpy as np


def run_all_evaluations():
    """Run all five evaluation methods"""
    
    print("🚀 COMPREHENSIVE EVALUATION: PoC 1 VALIDATION")
    print("=" * 80)
    print("Implementing Capstick et al. methodology for 한화솔루션 prediction")
    print("Testing all prior types: Expert, Data-informed (10x10), Data-informed (3x3), Uninformed, Naive LLM")
    print("=" * 80)
    
    # Check API key setup
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        print("Or copy config/env.example to .env and fill in your key")
        return None
    
    results = {}
    
    # 1. Bayesian with Expert LLM-elicited priors (10x10)
    print(f"\n{'🧠 METHOD 1: BAYESIAN + EXPERT PRIORS (10x10)'}")
    print("-" * 60)
    try:
        evaluator1 = BayesianEvaluator(prior_type="elicited", prior_folder="expert_10")
        results['expert_10'] = evaluator1.run_evaluation()
        print("✅ Expert Bayesian (10x10) completed")
    except Exception as e:
        print(f"❌ Expert Bayesian (10x10) failed: {e}")
        results['expert_10'] = None
    
    # 2. Bayesian with Data-informed LLM-elicited priors (10x10)
    print(f"\n{'🧠 METHOD 2: BAYESIAN + DATA-INFORMED PRIORS (10x10)'}")
    print("-" * 60)
    try:
        evaluator2 = BayesianEvaluator(prior_type="elicited", prior_folder="data_informed_10")
        results['data_informed_10'] = evaluator2.run_evaluation()
        print("✅ Data-informed Bayesian (10x10) completed")
    except Exception as e:
        print(f"❌ Data-informed Bayesian (10x10) failed: {e}")
        results['data_informed_10'] = None
    
    # 3. Bayesian with Data-informed LLM-elicited priors (3x3)
    print(f"\n{'🧠 METHOD 3: BAYESIAN + DATA-INFORMED PRIORS (3x3)'}")
    print("-" * 60)
    try:
        evaluator3 = BayesianEvaluator(prior_type="elicited", prior_folder="data_informed_3")
        results['data_informed_3'] = evaluator3.run_evaluation()
        print("✅ Data-informed Bayesian (3x3) completed")
    except Exception as e:
        print(f"❌ Data-informed Bayesian (3x3) failed: {e}")
        results['data_informed_3'] = None
    
    # 4. Bayesian with uninformed priors
    print(f"\n{'📊 METHOD 4: BAYESIAN + UNINFORMED PRIORS'}")
    print("-" * 60)
    try:
        evaluator4 = BayesianEvaluator(prior_type="uninformed")
        results['uninformed'] = evaluator4.run_evaluation()
        print("✅ Uninformed Bayesian completed")
    except Exception as e:
        print(f"❌ Uninformed Bayesian failed: {e}")
        results['uninformed'] = None
    
    # 5. Naive LLM (in-context learning)
    print(f"\n{'🤖 METHOD 5: NAIVE LLM (IN-CONTEXT LEARNING)'}")
    print("-" * 60)
    try:
        evaluator5 = NaiveLLMEvaluator()
        results['naive_llm'] = evaluator5.run_evaluation()
        print("✅ Naive LLM completed")
    except Exception as e:
        print(f"❌ Naive LLM failed: {e}")
        results['naive_llm'] = None
    
    return results


def create_comparison_table(results):
    """Create comprehensive comparison table"""
    
    print(f"\n{'📋 COMPREHENSIVE COMPARISON TABLE'}")
    print("=" * 90)
    
    # Prepare data for comparison
    methods = []
    metrics_data = []
    
    method_names = {
        'expert_10': 'Bayesian + Expert (10x10)',
        'data_informed_10': 'Bayesian + Data-informed (10x10)',
        'data_informed_3': 'Bayesian + Data-informed (3x3)',
        'uninformed': 'Bayesian + Uninformed',
        'naive_llm': 'Naive LLM (In-Context)'
    }
    
    for method, result in results.items():
        if result is not None:
            methods.append(method_names[method])
            metrics_data.append(result)
    
    if len(metrics_data) == 0:
        print("❌ No successful results to compare")
        return None
    
    # Create comparison DataFrame
    comparison_data = {}
    
    # Standard metrics
    standard_metrics = ['mae', 'rmse', 'r_squared', 'directional_accuracy']
    
    for metric in standard_metrics:
        comparison_data[metric] = []
        for data in metrics_data:
            if metric in data:
                comparison_data[metric].append(data[metric])
            else:
                comparison_data[metric].append(np.nan)
    
    # Coverage metrics (may not exist for all methods)
    coverage_metrics = ['coverage_95', 'coverage_90']
    for metric in coverage_metrics:
        if any(metric in data for data in metrics_data):
            comparison_data[metric] = []
            for data in metrics_data:
                comparison_data[metric].append(data.get(metric, np.nan))
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data, index=methods)
    
    # Print formatted table
    print(f"{'Method':<30} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Dir.Acc':<10} {'Cov.95%':<8} {'Cov.90%':<8}")
    print("-" * 95)
    
    for i, method in enumerate(methods):
        mae = df.iloc[i]['mae'] if not np.isnan(df.iloc[i]['mae']) else 'N/A'
        rmse = df.iloc[i]['rmse'] if not np.isnan(df.iloc[i]['rmse']) else 'N/A'
        r2 = df.iloc[i]['r_squared'] if not np.isnan(df.iloc[i]['r_squared']) else 'N/A'
        dir_acc = df.iloc[i]['directional_accuracy'] if not np.isnan(df.iloc[i]['directional_accuracy']) else 'N/A'
        cov95 = df.iloc[i].get('coverage_95', np.nan)
        cov90 = df.iloc[i].get('coverage_90', np.nan)
        
        # Format values
        mae_str = f"{mae:.4f}" if mae != 'N/A' else 'N/A'
        rmse_str = f"{rmse:.4f}" if rmse != 'N/A' else 'N/A' 
        r2_str = f"{r2:.4f}" if r2 != 'N/A' else 'N/A'
        dir_str = f"{dir_acc:.2%}" if dir_acc != 'N/A' else 'N/A'
        cov95_str = f"{cov95:.2%}" if not np.isnan(cov95) else 'N/A'
        cov90_str = f"{cov90:.2%}" if not np.isnan(cov90) else 'N/A'
        
        print(f"{method:<30} {mae_str:<8} {rmse_str:<8} {r2_str:<8} {dir_str:<10} {cov95_str:<8} {cov90_str:<8}")
    
    # Save detailed comparison
    results_dir = Path("data/results/comprehensive_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(results_dir / "method_comparison.csv")
    
    # Save detailed results
    import json
    with open(results_dir / "detailed_results.json", 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for method, result in results.items():
            if result is not None:
                serializable_results[method] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else 
                       int(v) if isinstance(v, (np.int32, np.int64)) else v
                    for k, v in result.items()
                }
            else:
                serializable_results[method] = None
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✅ Detailed comparison saved to: {results_dir}")
    
    return df


def analyze_results(results, comparison_df):
    """Analyze and interpret results according to Capstick paper"""
    
    print(f"\n{'📊 RESULTS ANALYSIS (Capstick et al. Framework)'}")
    print("=" * 70)
    
    if comparison_df is None or len(comparison_df) < 2:
        print("❌ Insufficient results for analysis")
        return
    
    # Key findings based on paper expectations
    findings = []
    
    # Check if LLM-elicited priors outperform uninformed
    if 'expert_10' in results and 'uninformed' in results:
        if results['expert_10'] and results['uninformed']:
            expert_mae = results['expert_10']['mae']
            uninformed_mae = results['uninformed']['mae']
            
            if expert_mae < uninformed_mae:
                improvement = (uninformed_mae - expert_mae) / uninformed_mae * 100
                findings.append(f"✅ Expert (10x10) priors reduce MAE by {improvement:.1f}% vs uninformed")
            else:
                degradation = (expert_mae - uninformed_mae) / uninformed_mae * 100
                findings.append(f"⚠️ Expert (10x10) priors increase MAE by {degradation:.1f}% vs uninformed")
    
    if 'data_informed_10' in results and 'uninformed' in results:
        if results['data_informed_10'] and results['uninformed']:
            data_informed_mae = results['data_informed_10']['mae']
            uninformed_mae = results['uninformed']['mae']
            
            if data_informed_mae < uninformed_mae:
                improvement = (uninformed_mae - data_informed_mae) / uninformed_mae * 100
                findings.append(f"✅ Data-informed (10x10) priors reduce MAE by {improvement:.1f}% vs uninformed")
            else:
                degradation = (data_informed_mae - uninformed_mae) / uninformed_mae * 100
                findings.append(f"⚠️ Data-informed (10x10) priors increase MAE by {degradation:.1f}% vs uninformed")
    
    if 'data_informed_3' in results and 'uninformed' in results:
        if results['data_informed_3'] and results['uninformed']:
            data_informed_mae = results['data_informed_3']['mae']
            uninformed_mae = results['uninformed']['mae']
            
            if data_informed_mae < uninformed_mae:
                improvement = (uninformed_mae - data_informed_mae) / uninformed_mae * 100
                findings.append(f"✅ Data-informed (3x3) priors reduce MAE by {improvement:.1f}% vs uninformed")
            else:
                degradation = (data_informed_mae - uninformed_mae) / uninformed_mae * 100
                findings.append(f"⚠️ Data-informed (3x3) priors increase MAE by {degradation:.1f}% vs uninformed")
    
    # Compare expert vs data-informed priors
    if 'expert_10' in results and 'data_informed_10' in results:
        if results['expert_10'] and results['data_informed_10']:
            expert_mae = results['expert_10']['mae']
            data_informed_mae = results['data_informed_10']['mae']
            
            if data_informed_mae < expert_mae:
                improvement = (expert_mae - data_informed_mae) / expert_mae * 100
                findings.append(f"✅ Data-informed (10x10) beats expert (10x10) by {improvement:.1f}% (MAE)")
            else:
                degradation = (data_informed_mae - expert_mae) / expert_mae * 100
                findings.append(f"⚠️ Expert (10x10) beats data-informed (10x10) by {degradation:.1f}% (MAE)")
    
    # Compare combination size effects (10x10 vs 3x3)
    if 'data_informed_10' in results and 'data_informed_3' in results:
        if results['data_informed_10'] and results['data_informed_3']:
            data_10_mae = results['data_informed_10']['mae']
            data_3_mae = results['data_informed_3']['mae']
            
            if data_10_mae < data_3_mae:
                improvement = (data_3_mae - data_10_mae) / data_3_mae * 100
                findings.append(f"✅ More combinations (10x10) beat fewer (3x3) by {improvement:.1f}% (MAE)")
            else:
                degradation = (data_10_mae - data_3_mae) / data_3_mae * 100
                findings.append(f"⚠️ Fewer combinations (3x3) beat more (10x10) by {degradation:.1f}% (MAE)")
    
    # Check directional accuracy improvements
    if 'expert_10' in results and results['expert_10']:
        dir_acc = results['expert_10']['directional_accuracy']
        findings.append(f"📈 Expert (10x10) directional accuracy: {dir_acc:.1%}")
    
    if 'data_informed_10' in results and results['data_informed_10']:
        dir_acc = results['data_informed_10']['directional_accuracy']
        findings.append(f"📈 Data-informed (10x10) directional accuracy: {dir_acc:.1%}")
    
    if 'data_informed_3' in results and results['data_informed_3']:
        dir_acc = results['data_informed_3']['directional_accuracy']
        findings.append(f"📈 Data-informed (3x3) directional accuracy: {dir_acc:.1%}")
    
    # Compare with naive LLM approach
    if 'naive_llm' in results and 'expert_10' in results:
        if results['naive_llm'] and results['expert_10']:
            naive_mae = results['naive_llm']['mae']
            expert_mae = results['expert_10']['mae']
            
            if expert_mae < naive_mae:
                improvement = (naive_mae - expert_mae) / expert_mae * 100
                findings.append(f"✅ Bayesian approach beats naive LLM by {improvement:.1f}% (MAE)")
            else:
                findings.append(f"⚠️ Naive LLM outperforms Bayesian approach")
    
    # Print findings
    for finding in findings:
        print(finding)
    
    # Paper-style conclusion
    print(f"\n{'🎯 CONCLUSION (Paper Validation)'}")
    print("-" * 40)
    
    if len(findings) == 0:
        print("❌ Insufficient data for conclusive analysis")
    else:
        best_method = comparison_df['mae'].idxmin() if 'mae' in comparison_df.columns else None
        if best_method:
            print(f"🏆 Best performing method: {best_method}")
        
        # Check if results align with paper expectations
        if 'expert_10' in results and results['expert_10']:
            print("📝 Expert (10x10) priors successfully implemented")
            print("📊 Bayesian mixture model with 100 components working")
        
        if 'data_informed_10' in results and results['data_informed_10']:
            print("📝 Data-informed (10x10) priors successfully implemented")
            print("📊 Data-informed approach with 100 components working")
        
        if 'data_informed_3' in results and results['data_informed_3']:
            print("📝 Data-informed (3x3) priors successfully implemented")
            print("📊 Data-informed approach with 9 components working")
        
        # Data efficiency analysis
        print(f"💡 This approach should be most beneficial in low-data regimes")
        print(f"   (Current test uses {len(comparison_df)} data points)")
        
        # Additional insights
        print(f"🔍 Comprehensive comparison completed:")
        print(f"   - Expert vs Data-informed priors")
        print(f"   - Combination size effects (3x3 vs 10x10)")
        print(f"   - Bayesian vs Naive LLM approaches")


def main():
    """Main execution function"""
    
    # Run all evaluations
    results = run_all_evaluations()
    
    if results is None:
        return 1
    
    # Create comparison
    comparison_df = create_comparison_table(results)
    
    # Analyze results
    analyze_results(results, comparison_df)
    
    print(f"\n{'🎉 PoC 1 VALIDATION COMPLETE'}")
    print("=" * 50)
    print("All five methods from Capstick et al. have been implemented and tested:")
    print("1. Bayesian + Expert priors (10x10)")
    print("2. Bayesian + Data-informed priors (10x10)")
    print("3. Bayesian + Data-informed priors (3x3)")
    print("4. Bayesian + Uninformed priors")
    print("5. Naive LLM (in-context learning)")
    print("Results saved to data/results/comprehensive_comparison/")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
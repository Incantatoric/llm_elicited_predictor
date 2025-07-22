#!/usr/bin/env python3
"""
Debug script to test feature matching logic in _process_prior_response
Uses real data from raw_priors.json to isolate the bug
"""

import json
import numpy as np
from difflib import SequenceMatcher

def test_feature_matching():
    """Test the exact feature matching logic from _process_prior_response"""
    
    print("🔍 DEBUGGING FEATURE MATCHING LOGIC")
    print("=" * 50)
    
    # Load real data
    with open("data/priors/expert_10/raw_priors.json", 'r') as f:
        raw_priors = json.load(f)
    
    # Expected feature names (from the model)
    feature_names = ['kospi_return', 'oil_price_change', 'usd_krw_change', 'vix_change', 'materials_sector_return']
    
    print(f"✓ Loaded {len(raw_priors)} raw priors")
    print(f"✓ Expected features: {feature_names}")
    print()
    
    # Test first 3 priors
    for i in range(min(3, len(raw_priors))):
        print(f"🔬 TESTING PRIOR {i}")
        print("-" * 30)
        
        prior_data = raw_priors[i]
        print(f"LLM provided keys: {list(prior_data.keys())}")
        print()
        
        # Test exact matching first
        print("  Testing exact matching:")
        for feature in feature_names:
            if feature in prior_data:
                print(f"    ✅ EXACT MATCH: {feature}")
                feature_data = prior_data[feature]
                print(f"      Data: {feature_data}")
            else:
                print(f"    ❌ NO EXACT MATCH: {feature}")
        
        print()
        
        # Test fuzzy matching
        print("  Testing fuzzy matching:")
        for feature in feature_names:
            if feature in prior_data:
                print(f"    Skipping {feature} (exact match found)")
                continue
                
            best_match = None
            highest_similarity = 0
            
            for response_feature in prior_data.keys():
                similarity = SequenceMatcher(None, feature, response_feature).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = response_feature
            
            print(f"    {feature}: best_match='{best_match}', similarity={highest_similarity:.3f}")
            
            if best_match and highest_similarity > 0.3:
                print(f"      ✅ FUZZY MATCH: {feature} -> {best_match}")
                feature_data = prior_data[best_match]
                print(f"      Data: {feature_data}")
            else:
                print(f"      ❌ NO FUZZY MATCH: {feature}")
        
        print()
        
        # Test the actual extraction logic
        print("  Testing value extraction:")
        feature_priors = []
        
        for feature in feature_names:
            # Try exact match first
            if feature in prior_data:
                feature_data = prior_data[feature]
                print(f"    {feature}: EXACT MATCH")
            else:
                # Fuzzy matching
                best_match = None
                highest_similarity = 0
                
                for response_feature in prior_data.keys():
                    similarity = SequenceMatcher(None, feature, response_feature).ratio()
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = response_feature
                
                if best_match and highest_similarity > 0.3:
                    feature_data = prior_data[best_match]
                    print(f"    {feature}: FUZZY MATCH -> {best_match}")
                else:
                    print(f"    {feature}: ❌ NO MATCH FOUND")
                    print(f"      Available keys: {list(prior_data.keys())}")
                    continue
            
            # Extract mean and std
            if "mean" not in feature_data:
                print(f"      ❌ NO 'mean' key in {feature_data}")
                continue
            if "std" not in feature_data:
                print(f"      ❌ NO 'std' key in {feature_data}")
                continue
            
            mean_val = float(feature_data["mean"])
            std_val = max(float(feature_data["std"]), 1e-3)
            
            feature_priors.append([mean_val, std_val])
            print(f"      ✅ EXTRACTED: mean={mean_val:.3f}, std={std_val:.3f}")
        
        print(f"  Final feature_priors: {feature_priors}")
        print()

if __name__ == "__main__":
    test_feature_matching() 
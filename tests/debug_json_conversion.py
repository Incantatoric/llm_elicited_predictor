#!/usr/bin/env python3
"""
Debug script to test JSON to NPY conversion process
Uses actual raw_priors.json data to find the conversion bug
"""

import json
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path

def test_json_conversion():
    """Test the exact JSON→NPY conversion logic"""
    
    print("🔍 DEBUGGING JSON→NPY CONVERSION")
    print("=" * 50)
    
    # Load actual data
    with open("data/priors/expert_10/raw_priors.json", "r") as f:
        raw_priors = json.load(f)
    
    with open("data/processed/metadata.json", "r") as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    
    print(f"✓ Loaded {len(raw_priors)} raw priors")
    print(f"✓ Expected features: {feature_names}")
    
    # Test first few priors
    for i in range(min(3, len(raw_priors))):
        print(f"\n🔬 TESTING PRIOR {i}")
        print("-" * 30)
        
        prior_data = raw_priors[i]
        print(f"LLM provided keys: {list(prior_data.keys())}")
        
        # Replicate the exact conversion logic from _process_prior_response
        bias_prior = [0.0, 1.0]  # Uninformative bias
        feature_priors = []
        conversion_log = []
        
        for feature in feature_names:
            print(f"\n  Processing feature: {feature}")
            
            # Fuzzy matching (exact copy of main code)
            best_match = None
            highest_similarity = 0
            
            for response_feature in prior_data.keys():
                similarity = SequenceMatcher(None, feature, response_feature).ratio()
                print(f"    Similarity with '{response_feature}': {similarity:.3f}")
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = response_feature
            
            print(f"    Best match: '{best_match}' (similarity: {highest_similarity:.3f})")
            
            if best_match and highest_similarity > 0.3:
                feature_data = prior_data[best_match]
                print(f"    LLM data: {feature_data}")
                
                # Check for std key (exact copy of main code)
                std_key = 'std' if 'std' in feature_data else ('std_dev' if 'std_dev' in feature_data else None)
                
                if std_key is None:
                    print(f"    ❌ No std/std_dev key found! Using uninformative [0.0, 1.0]")
                    feature_priors.append([0.0, 1.0])
                    conversion_log.append(f"{feature}: NO_STD_KEY → [0.0, 1.0]")
                else:
                    std_val = max(float(feature_data[std_key]), 1e-3)
                    mean_val = float(feature_data["mean"])
                    feature_priors.append([mean_val, std_val])
                    conversion_log.append(f"{feature}: [{mean_val}, {std_val}]")
                    print(f"    ✅ Converted: mean={mean_val}, std={std_val}")
            else:
                print(f"    ❌ Match failed (similarity {highest_similarity:.3f} <= 0.3)! Using uninformative [0.0, 1.0]")
                feature_priors.append([0.0, 1.0])
                conversion_log.append(f"{feature}: MATCH_FAILED → [0.0, 1.0]")
        
        # Create final array
        full_prior = [bias_prior] + feature_priors
        prior_array = np.array(full_prior)
        
        print(f"\n📊 FINAL CONVERSION RESULT:")
        print(f"Final array shape: {prior_array.shape}")
        print(f"Final array:\n{prior_array}")
        
        print(f"\n📋 CONVERSION LOG:")
        for log_entry in conversion_log:
            print(f"  {log_entry}")
        
        # Compare with what we expect vs what we got
        print(f"\n🔍 COMPARISON:")
        expected_kospi = prior_data['kospi_return']
        actual_kospi = prior_array[1]  # kospi_return is first feature (index 1 after bias)
        
        print(f"Expected kospi_return: mean={expected_kospi['mean']}, std={expected_kospi['std']}")
        print(f"Actual NPY kospi_return: {actual_kospi}")
        
        if actual_kospi[0] != expected_kospi['mean'] or actual_kospi[1] != expected_kospi['std']:
            print("❌ MISMATCH FOUND!")
            break
        else:
            print("✅ Match successful")
    
    # Additional checks
    print(f"\n🔧 ADDITIONAL CHECKS:")
    
    # Test a real NPY file
    npy_file = "data/priors/expert_10/hanwha_prior_0.npy"
    if Path(npy_file).exists():
        saved_array = np.load(npy_file)
        print(f"Actual saved NPY file {npy_file}:")
        print(f"Shape: {saved_array.shape}")
        print(f"Content:\n{saved_array}")
        
        # Compare first feature (kospi_return)
        expected_from_json = raw_priors[0]['kospi_return'] 
        actual_from_npy = saved_array[1]  # kospi_return = index 1
        
        print(f"\n🎯 ROOT CAUSE ANALYSIS:")
        print(f"JSON says kospi_return should be: [{expected_from_json['mean']}, {expected_from_json['std']}]")
        print(f"NPY file actually contains: {actual_from_npy}")
        
        if (actual_from_npy[0] != expected_from_json['mean'] or 
            actual_from_npy[1] != expected_from_json['std']):
            print("❌ CONFIRMED: JSON→NPY conversion is broken!")
        else:
            print("✅ SURPRISE: JSON→NPY conversion is working correctly")

if __name__ == "__main__":
    test_json_conversion() 
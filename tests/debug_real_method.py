#!/usr/bin/env python3
"""
Debug script to test the ACTUAL _process_prior_response method from the codebase
"""

import json
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hanwha_predictor.elicitation.llm_elicitor import HanwhaLLMElicitor

def test_real_method():
    """Test the actual _process_prior_response method"""
    
    print("🔍 TESTING ACTUAL _process_prior_response METHOD")
    print("=" * 50)
    
    # Load real data
    with open("data/priors/expert_10/raw_priors.json", 'r') as f:
        raw_priors = json.load(f)
    
    # Expected feature names
    feature_names = ['kospi_return', 'oil_price_change', 'usd_krw_change', 'vix_change', 'materials_sector_return']
    
    print(f"✓ Loaded {len(raw_priors)} raw priors")
    print(f"✓ Expected features: {feature_names}")
    print()
    
    # Create elicitor instance (we don't need API key for this test)
    elicitor = HanwhaLLMElicitor(api_key="dummy", model_name="dummy")
    
    # Test first 3 priors
    for i in range(min(3, len(raw_priors))):
        print(f"🔬 TESTING PRIOR {i}")
        print("-" * 30)
        
        # Convert back to JSON string (as the method expects)
        result_text = json.dumps(raw_priors[i], indent=2)
        
        print(f"Input JSON: {result_text[:200]}...")
        print()
        
        try:
            # Call the actual method
            prior_array, reasoning = elicitor._process_prior_response(
                result_text, feature_names, i
            )
            
            print(f"✅ Method succeeded!")
            print(f"Output array shape: {prior_array.shape}")
            print(f"Output array:")
            print(prior_array)
            print()
            
            # Check if we got uninformed priors
            if np.allclose(prior_array[1:, 0], 0.0) and np.allclose(prior_array[1:, 1], 1.0):
                print("❌ PROBLEM: Got uninformed priors [0.0, 1.0] for all features!")
            else:
                print("✅ SUCCESS: Got diverse priors!")
                
        except Exception as e:
            print(f"❌ Method failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    test_real_method() 
#!/usr/bin/env python3
"""
Diagnostic script to investigate prior loading issue
Tests exactly what HanwhaBayesianModel.load_priors() does
"""

import numpy as np
import os
from pathlib import Path

def test_prior_loading():
    """Test prior loading exactly like HanwhaBayesianModel does"""
    
    priors_dir = "data/priors/expert_10"
    
    print("🔍 DIAGNOSING PRIOR LOADING ISSUE")
    print("=" * 50)
    
    # 1. Check files exist
    print(f"📁 Checking directory: {priors_dir}")
    prior_files = list(Path(priors_dir).glob("hanwha_prior_*.npy"))
    print(f"✓ Found {len(prior_files)} .npy files")
    
    # 2. Test loading exactly like main code
    print(f"\n📊 TESTING LOADING (exactly like HanwhaBayesianModel)")
    priors = []
    i = 0
    loaded_files = []
    failed_files = []
    
    while True:
        prior_file = f"{priors_dir}/hanwha_prior_{i}.npy"
        try:
            prior = np.load(prior_file)
            priors.append(prior)
            loaded_files.append(prior_file)
            print(f"✓ Loaded {prior_file}: shape {prior.shape}")
            
            # Show first few values for variety check
            if i < 5:
                print(f"   kospi_return mean/std: {prior[0]}")
            
            i += 1
        except FileNotFoundError:
            failed_files.append(prior_file)
            print(f"❌ Failed to load {prior_file} - STOPPING")
            break
        except Exception as e:
            print(f"❌ Error loading {prior_file}: {e}")
            failed_files.append(prior_file)
            break
    
    print(f"\n📋 LOADING SUMMARY:")
    print(f"✓ Successfully loaded: {len(loaded_files)} files")
    print(f"❌ Failed files: {len(failed_files)}")
    
    if len(priors) == 0:
        print("❌ NO PRIORS LOADED!")
        return
    
    # 3. Test array stacking (like main code)
    print(f"\n🔧 TESTING ARRAY STACKING")
    try:
        priors_array = np.array(priors)
        print(f"✓ Stacked array shape: {priors_array.shape}")
        print(f"   Expected shape: (100, 6, 2)")
        
        if priors_array.shape[0] != 100:
            print(f"⚠️  WARNING: Got {priors_array.shape[0]} priors instead of 100!")
        
    except Exception as e:
        print(f"❌ Error stacking arrays: {e}")
        # Check individual shapes
        print("🔍 Individual prior shapes:")
        for i, prior in enumerate(priors[:10]):  # First 10
            print(f"   Prior {i}: {prior.shape}")
        return
    
    # 4. Test visualization access (like DistributionVisualizer)
    print(f"\n📈 TESTING VISUALIZATION ACCESS")
    param_idx = 0  # kospi_return (first feature after intercept)
    try:
        means = priors_array[:, param_idx, 0]  # All means for kospi_return
        stds = priors_array[:, param_idx, 1]   # All stds for kospi_return
        
        print(f"✓ kospi_return means array length: {len(means)}")
        print(f"✓ kospi_return stds array length: {len(stds)}")
        
        # Check if values are actually different (not all identical)
        unique_means = len(np.unique(np.round(means, 4)))
        unique_stds = len(np.unique(np.round(stds, 4)))
        
        print(f"✓ Unique mean values: {unique_means}")
        print(f"✓ Unique std values: {unique_stds}")
        
        if unique_means < 10:
            print(f"⚠️  WARNING: Only {unique_means} unique means - priors might be too similar!")
            print(f"   First 10 means: {means[:10]}")
        
        if unique_stds < 5:
            print(f"⚠️  WARNING: Only {unique_stds} unique stds - priors might be too similar!")
            print(f"   First 10 stds: {stds[:10]}")
        
        # Show what visualizer would see for first 20 components
        print(f"\n🎨 WHAT VISUALIZER SEES (first 20 components):")
        display_means = means[:20]
        display_stds = stds[:20]
        print(f"   Means (first 20): {display_means}")
        print(f"   Stds (first 20): {display_stds}")
        
        # This would be the scatter plot data
        unique_display_means = len(np.unique(np.round(display_means, 4)))
        unique_display_stds = len(np.unique(np.round(display_stds, 4)))
        print(f"   Unique display means: {unique_display_means}")
        print(f"   Unique display stds: {unique_display_stds}")
        
        if unique_display_means == 2 and unique_display_stds == 2:
            print("🎯 FOUND THE ISSUE: First 20 priors collapse to only 2 unique values!")
            print("   This explains why you see only 2 points in the scatter plot!")
    
    except Exception as e:
        print(f"❌ Error in visualization access: {e}")
    
    # 5. Additional diagnostics
    print(f"\n🔍 ADDITIONAL DIAGNOSTICS:")
    
    # Check file ordering
    first_5_files = [f"hanwha_prior_{i}.npy" for i in range(5)]
    for filename in first_5_files:
        filepath = Path(priors_dir) / filename
        if filepath.exists():
            print(f"✓ {filename} exists")
        else:
            print(f"❌ {filename} MISSING - This could cause early termination!")
    
    print(f"\n🎯 CONCLUSION:")
    if len(priors) < 100:
        print(f"❌ LOADING ISSUE: Only loaded {len(priors)}/100 priors")
        print("   This explains the mixture component issue!")
    elif unique_means < 10:
        print(f"❌ SIMILARITY ISSUE: Priors are too similar ({unique_means} unique values)")
        print("   LLM might be generating nearly identical priors!")
    else:
        print("✅ Loading appears correct - issue might be elsewhere")


if __name__ == "__main__":
    test_prior_loading() 
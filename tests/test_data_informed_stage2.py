#!/usr/bin/env python3
"""
Test script for Stage 2: Data-informed prior generation (self-contained)

This script demonstrates that DataInformedElicitor now works independently
without requiring the original elicitor's output files.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_stage_2_ready():
    """Test that Stage 2 is ready to run independently"""
    print("=" * 60)
    print("STAGE 2 INDEPENDENCE TEST")
    print("=" * 60)
    
    # Check API key
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    print(f"• API Key set: {'✅' if api_key_set else '❌'}")
    
    if not api_key_set:
        print("❌ OPENAI_API_KEY environment variable not set")
        return False
    
    # Test DataInformedElicitor import
    try:
        from src.hanwha_predictor.elicitation.data_informed_elicitor import DataInformedElicitor
        print("✅ DataInformedElicitor imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test class initialization with real API key
    try:
        elicitor = DataInformedElicitor(api_key=os.getenv("OPENAI_API_KEY"))
        print("✅ DataInformedElicitor initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test data loading (should work without any dependencies)
    try:
        X_train, y_train, feature_names = elicitor.load_training_data()
        print(f"✅ Training data loaded: {X_train.shape[0]} months, {len(feature_names)} features")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Test context preparation
    try:
        data_context = elicitor.prepare_data_context(X_train, y_train)
        print(f"✅ Data context prepared: {len(data_context)} characters")
    except Exception as e:
        print(f"❌ Context preparation failed: {e}")
        return False
    
    # Test prompt creation 
    try:
        base_system, base_user = elicitor.create_data_informed_prompts(feature_names, data_context)
        print(f"✅ Data-informed prompts created")
        print(f"   System prompt: {len(base_system)} chars")
        print(f"   User prompt: {len(base_user)} chars")
    except Exception as e:
        print(f"❌ Prompt creation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("STAGE 2: READY TO EXECUTE ✅")
    print("DataInformedElicitor is now self-contained and can run independently!")
    print("No dependencies on original elicitor output files.")
    print("=" * 60)
    
    return True

def show_execution_options():
    """Show different ways to execute Stage 2"""
    print("\n" + "=" * 60)
    print("STAGE 2 EXECUTION OPTIONS")
    print("=" * 60)
    
    print("Option 1: Test with 9 combinations (3x3) - Fast")
    print("   python -c \"")
    print("   from src.hanwha_predictor.elicitation.data_informed_elicitor import DataInformedElicitor")
    print("   import os")
    print("   elicitor = DataInformedElicitor(os.getenv('OPENAI_API_KEY'))")
    print("   priors = elicitor.elicit_data_informed_priors(n_combinations=9)")
    print("   print(f'Generated {len(priors)} data-informed priors with shape {priors.shape}')\"")
    print()
    
    print("Option 2: Full 100 combinations (10x10) - Complete")
    print("   python -c \"")
    print("   from src.hanwha_predictor.elicitation.data_informed_elicitor import DataInformedElicitor")
    print("   import os")
    print("   elicitor = DataInformedElicitor(os.getenv('OPENAI_API_KEY'))")  
    print("   priors = elicitor.elicit_data_informed_priors(n_combinations=100)")
    print("   print(f'Generated {len(priors)} data-informed priors with shape {priors.shape}')\"")
    print()
    
    print("Option 3: Direct script execution")
    print("   python src/hanwha_predictor/elicitation/data_informed_elicitor.py")
    print()
    
    print("Expected outputs:")
    print("• config/prompts/data_informed/system_roles_data_informed.txt")
    print("• config/prompts/data_informed/user_roles_data_informed.txt")
    print("• data/priors/data_informed_9/ (for 9 combinations)")
    print("• data/priors/data_informed_100/ (for 100 combinations)")
    print("• Detailed reasoning data and summaries")
    
    print("=" * 60)

if __name__ == "__main__":
    # Test Stage 2 readiness
    stage_2_ready = test_stage_2_ready()
    
    if stage_2_ready:
        show_execution_options()
        print("\n🎉 Stage 2 Ready - DataInformedElicitor is self-contained!")
        print("You can now run data-informed prior generation independently!")
    else:
        print("\n❌ Stage 2 Not Ready - Fix issues before proceeding") 
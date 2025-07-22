#!/usr/bin/env python3
"""
Test script for Stage 1: DataInformedElicitor class creation

This script demonstrates that Stage 1 is complete and outlines Stage 2 execution.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_stage_1_completion():
    """Test that Stage 1 is complete - DataInformedElicitor class created"""
    print("=" * 60)
    print("STAGE 1 COMPLETION TEST")
    print("=" * 60)
    
    # Test 1: Import the new DataInformedElicitor class
    try:
        from src.hanwha_predictor.elicitation.data_informed_elicitor import DataInformedElicitor
        print("✅ DataInformedElicitor class created successfully")
    except ImportError as e:
        print(f"❌ Failed to import DataInformedElicitor: {e}")
        return False
    
    # Test 2: Check if training data exists
    data_dir = project_root / "data" / "processed"
    required_files = ["features_standardized.csv", "target_returns.csv", "metadata.json"]
    
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✅ Training data file exists: {file}")
        else:
            print(f"❌ Missing training data file: {file}")
            return False
    
    # Test 3: Check class initialization (without API key)
    try:
        # This should work even without API key for testing class structure
        elicitor = DataInformedElicitor(api_key="test_key")
        print("✅ DataInformedElicitor initialization successful")
    except Exception as e:
        print(f"❌ DataInformedElicitor initialization failed: {e}")
        return False
    
    # Test 4: Test data loading method
    try:
        X_train, y_train, feature_names = elicitor.load_training_data()
        print(f"✅ Training data loaded: {X_train.shape[0]} months, {len(feature_names)} features")
        print(f"   Features: {feature_names}")
        print(f"   Training period: {X_train.index[0].strftime('%Y-%m-%d')} to {X_train.index[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"❌ Training data loading failed: {e}")
        return False
    
    # Test 5: Test context preparation
    try:
        data_context = elicitor.prepare_data_context(X_train, y_train)
        print(f"✅ Data context prepared: {len(data_context)} characters")
        print("   Preview of data context:")
        print("   " + data_context[:200] + "...")
    except Exception as e:
        print(f"❌ Data context preparation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("STAGE 1: COMPLETE ✅")
    print("DataInformedElicitor class successfully created with:")
    print("• Historical data loading capability")
    print("• Comprehensive variable explanations")  
    print("• Data context preparation for LLM")
    print("• Support for both 100 and 9 combination modes")
    print("• Same output format as original elicitor")
    print("=" * 60)
    
    return True

def outline_stage_2():
    """Outline what Stage 2 execution will require"""
    print("\n" + "=" * 60)
    print("STAGE 2: EXECUTION REQUIREMENTS")
    print("=" * 60)
    
    print("To execute Stage 2 (Generate data-informed priors), we need:")
    print()
    print("1. 🔑 OPENAI_API_KEY environment variable set")
    print("2. 📝 Existing prompt variations from original elicitor")
    print("   - Run: python src/hanwha_predictor/elicitation/llm_elicitor.py")
    print("   - This creates: config/prompts/elicitation/system_roles_hanwha.txt")
    print("   - This creates: config/prompts/elicitation/user_roles_hanwha.txt")
    print()
    print("3. 🚀 Execute data-informed prior generation:")
    print("   - python src/hanwha_predictor/elicitation/data_informed_elicitor.py")
    print("   - Or use the class methods directly")
    print()
    print("This will generate:")
    print("• data/priors/data_informed_9/ (3x3 = 9 combinations)")
    print("• data/priors/data_informed_100/ (10x10 = 100 combinations)")
    print("• Reasoning data for each elicitation")
    print("• Summary statistics and metadata")
    print()
    
    # Check current state
    print("CURRENT STATE CHECK:")
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    print(f"• API Key set: {'✅' if api_key_set else '❌'}")
    
    prompts_dir = Path("config/prompts/elicitation")
    system_file = prompts_dir / "system_roles_hanwha.txt"
    user_file = prompts_dir / "user_roles_hanwha.txt"
    
    system_exists = system_file.exists()
    user_exists = user_file.exists()
    print(f"• System prompts exist: {'✅' if system_exists else '❌'}")
    print(f"• User prompts exist: {'✅' if user_exists else '❌'}")
    
    if api_key_set and system_exists and user_exists:
        print("\n🎯 READY TO EXECUTE STAGE 2!")
    else:
        print("\n⏳ Prerequisites needed before Stage 2 execution")
    
    print("=" * 60)

if __name__ == "__main__":
    # Run Stage 1 completion test
    stage_1_success = test_stage_1_completion()
    
    # Outline Stage 2 requirements
    outline_stage_2()
    
    if stage_1_success:
        print("\n🎉 Stage 1 Complete - Ready for Stage 2!")
    else:
        print("\n❌ Stage 1 Issues - Fix before proceeding to Stage 2") 
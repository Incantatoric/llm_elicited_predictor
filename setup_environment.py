#!/usr/bin/env python3
"""
Environment setup script for LLM Elicited Predictor
Helps configure API keys and dependencies properly
"""

import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'pymc', 'arviz', 'openai', 'yfinance', 'scipy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: poetry install")
        return False
    else:
        print("✅ All dependencies installed")
        return True


def check_api_key():
    """Check if OpenAI API key is configured"""
    print("\n🔑 Checking API key configuration...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("\n📋 Setup Instructions:")
        print("1. Copy config/env.example to .env in project root:")
        print("   cp config/env.example .env")
        print("2. Edit .env and add your OpenAI API key:")
        print("   OPENAI_API_KEY=your-actual-key-here")
        print("3. Load environment variables:")
        print("   source .env  # or use python-dotenv")
        return False
    
    # Check if it looks like a real key (starts with sk-)
    if not api_key.startswith('sk-'):
        print("⚠️ API key found but doesn't look like OpenAI format (should start with 'sk-')")
        return False
    
    print("✅ API key configured correctly")
    return True


def check_data_files():
    """Check if required data files exist"""
    print("\n📁 Checking data files...")
    
    required_files = [
        'data/processed/features_standardized.csv',
        'data/processed/target_returns.csv',
        'data/processed/metadata.json'
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"❌ Missing data files: {', '.join(missing)}")
        print("Run data collection first: python main.py")
        return False
    else:
        print("✅ All data files present")
        return True


def check_prior_files():
    """Check if LLM-elicited prior files exist"""
    print("\n🧠 Checking LLM prior files...")
    
    priors_dir = Path('data/priors/expert_10')
    if not priors_dir.exists():
        print(f"❌ Priors directory not found: {priors_dir}")
        print("Run prior elicitation first: python main.py")
        return False
    
    # Check for numbered prior files
    prior_files = list(priors_dir.glob('hanwha_prior_*.npy'))
    
    if len(prior_files) < 10:
        print(f"⚠️ Only {len(prior_files)} prior files found (expected ~100)")
        print("Run prior elicitation: python main.py")
        return False
    
    print(f"✅ Found {len(prior_files)} prior files")
    return True


def setup_environment():
    """Complete environment setup check"""
    print("🚀 LLM ELICITED PREDICTOR - ENVIRONMENT SETUP")
    print("=" * 60)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("API Key", check_api_key),
        ("Data Files", check_data_files), 
        ("Prior Files", check_prior_files)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results[name] = False
    
    print(f"\n{'🎯 SETUP SUMMARY'}")
    print("-" * 30)
    
    for name, passed in results.items():
        status = "✅ READY" if passed else "❌ NEEDS SETUP"
        print(f"{name:<15} {status}")
    
    all_ready = all(results.values())
    
    if all_ready:
        print(f"\n🎉 ENVIRONMENT READY!")
        print("You can now run:")
        print("  python poc1_validation/compare_all_methods.py")
        return True
    else:
        print(f"\n⚠️ SETUP INCOMPLETE")
        print("Please fix the issues above before running evaluations")
        return False


if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1) 
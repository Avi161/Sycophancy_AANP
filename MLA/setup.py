#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for sycophancy mitigation experiments
Run this first to check dependencies and setup
"""

import os
import sys
import subprocess
import importlib

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'torch', 'transformers', 'transformer_lens', 'einops', 'jaxtyping', 
        'colorama', 'scikit-learn', 'datasets', 'tqdm', 'huggingface_hub', 
        'requests', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    
    print("Checking dependencies...")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them with:")
        print("pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies satisfied!")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (much slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def setup_sycophancy_eval():
    """Clone the sycophancy evaluation repository"""
    if os.path.exists("sycophancy-eval"):
        print("✅ sycophancy-eval repository already exists")
        return True
    
    print("📦 Cloning sycophancy-eval repository...")
    try:
        result = subprocess.run(
            ["git", "clone", "https://github.com/meg-tong/sycophancy-eval.git"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ sycophancy-eval cloned successfully")
            return True
        else:
            print(f"❌ Failed to clone repository: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Git not found. Please install Git or manually clone:")
        print("   git clone https://github.com/meg-tong/sycophancy-eval.git")
        return False

def check_huggingface_token():
    """Check if HuggingFace token is configured"""
    print("\n🤗 HuggingFace Token Check:")
    print("The code requires a HuggingFace token to access the Gemma model.")
    print("Please ensure you have:")
    print("1. A HuggingFace account with access to google/gemma-2-2b-it")
    print("2. Updated the token in model_utils.py line 12")
    print("3. Or set it as environment variable: export HF_TOKEN=your_token")
    
    # Check if token exists in environment
    if os.getenv('HF_TOKEN'):
        print("✅ HF_TOKEN environment variable found")
    else:
        print("⚠️  No HF_TOKEN environment variable - make sure token is in model_utils.py")

def main():
    """Main setup function"""
    print("="*60)
    print("🚀 SYCOPHANCY MITIGATION SETUP")
    print("="*60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check CUDA
    cuda_ok = check_cuda()
    
    # Setup evaluation data
    eval_ok = setup_sycophancy_eval()
    
    # Check HuggingFace token
    check_huggingface_token()
    
    print("\n" + "="*60)
    print("📋 SETUP SUMMARY")
    print("="*60)
    print(f"Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"CUDA: {'✅' if cuda_ok else '⚠️ '}")
    print(f"Evaluation Data: {'✅' if eval_ok else '❌'}")
    
    if deps_ok and eval_ok:
        print("\n🎉 Setup complete! You can now run:")
        print("   python main.py")
        print("\nFor step-by-step guidance, see README.md")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above before proceeding.")
        return False
    
    return True

if __name__ == "__main__":
    main()
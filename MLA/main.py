# -*- coding: utf-8 -*-
"""
Main execution script for sycophancy mitigation experiments
"""

import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc
import json
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable, Dict, Tuple
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore

# Import our organized modules
from model_utils import setup_model
from data_utils import load_sycophancy_data
from direction_finding import find_sycophancy_directions_fixed
from evaluation import evaluate_anti_sycophancy_intervention_fixed
from interactive import interactive_dataset_evaluation, quick_dataset_eval
from generation import _generate_responses
from intervention import _generate_with_anti_sycophancy_hooks_fixed
from config import DEVICE
from storage import save_directions_simple, load_directions_simple, list_saved_directions

def setup_global_vars_after_loading(data):
    """Set up global variables after loading data"""
    global filtered_directions, layer_directions, model, stats
    
    filtered_directions = data['filtered_directions']
    layer_directions = data['layer_directions'] 
    model = setup_model()
    stats = data['stats']
    
    # Make sure they're also in globals() for the interactive module
    globals()['filtered_directions'] = filtered_directions
    globals()['layer_directions'] = layer_directions
    globals()['model'] = model
    globals()['stats'] = stats
    globals()['_generate_responses'] = _generate_responses
    globals()['_generate_with_anti_sycophancy_hooks_fixed'] = _generate_with_anti_sycophancy_hooks_fixed
    globals()['DEVICE'] = DEVICE
    
    print("Global variables set up successfully!")

def main_fixed():
    """FIXED main execution function with better parameters"""

    # Load sycophancy data
    print("Loading sycophancy dataset...")
    jsonl_path = "sycophancy-eval/datasets/answer.jsonl"
    data = load_sycophancy_data(jsonl_path, test_ratio=0.2)

    # Setup model
    model = setup_model()

    # Find anti-sycophancy directions with FIXED method
    print("Finding anti-sycophancy directions...")
    layer_directions, stats = find_sycophancy_directions_fixed(
        model,
        data['train_groups'],
        n_samples=100,  # Reduced for stability
        batch_size=1,   # Smaller batch size
    )

    if not layer_directions:
        print("No valid directions found! Check your data and method.")
        return None, None, None, None

    # Filter layers more carefully
    min_effect_size = 0.4  # Higher threshold
    filtered_directions = {
        layer: direction for layer, direction in layer_directions.items()
        if stats['layer_stats'][layer]['effect_size'] >= min_effect_size
    }

    print(f"\nFiltered to {len(filtered_directions)}/{len(layer_directions)} layers with effect size >= {min_effect_size}")

    if not filtered_directions:
        print("No layers meet threshold. Using top 3 layers by effect size.")
        sorted_layers = sorted(
            layer_directions.keys(),
            key=lambda x: stats['layer_stats'][x]['effect_size'],
            reverse=True
        )[:3]
        filtered_directions = {layer: layer_directions[layer] for layer in sorted_layers}
        print("Using layers:", sorted_layers)
    else:
        print("Using layers:", sorted(filtered_directions.keys()))

    # SAVE THE DIRECTIONS
    print("\nSaving directions for later use...")
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    experiment_name = f"gemma2_2b_sycophancy_{timestamp}"
    
    saved_name = save_directions_simple(layer_directions, filtered_directions, stats, experiment_name)

    # Evaluate with more conservative parameters
    print("Evaluating anti-sycophancy intervention...")
    alpha = 0.2  # Very conservative alpha
    results = evaluate_anti_sycophancy_intervention_fixed(
        model,
        data['test_groups'],
        filtered_directions,
        alpha,
        n_test=10,
        debug=True
    )

    return layer_directions, filtered_directions, stats, results, saved_name


# MAIN EXECUTION
if __name__ == "__main__":
    # Clone the evaluation repository if it doesn't exist
    import os
    if not os.path.exists("sycophancy-eval"):
        print("Cloning sycophancy evaluation repository...")
        os.system("git clone https://github.com/meg-tong/sycophancy-eval.git")
    
    # Check if user wants to load existing experiment
    print("\n" + "="*80)
    print("SYCOPHANCY MITIGATION EXPERIMENT")
    print("="*80)
    print("Options:")
    print("1. Run new experiment (compute directions from scratch)")
    print("2. Load saved directions for evaluation")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    
    if choice == "2":
        list_saved_directions()
        exp_name = input("\nEnter experiment filename (without .pt): ").strip()
        
        data = load_directions_simple(exp_name)
        if data is not None:
            setup_global_vars_after_loading(data)
            
            print("\n" + "="*80)
            print("LOADED SAVED EXPERIMENT")
            print("="*80)
            print("Ready for evaluation!")
            print(f"Using {len(data['filtered_directions'])} layers: {sorted(data['filtered_directions'].keys())}")
            
            user_choice = input("\nStart interactive evaluation now? (y/N): ").strip().lower()
            if user_choice in ['y', 'yes']:
                interactive_dataset_evaluation()
        else:
            print("Failed to load experiment")
        exit()
    
    # Run new experiment
    results = main_fixed()
    
    if results[0] is not None:  # If successful
        layer_directions, filtered_directions, stats, res, saved_name = results
        
        # FIXED: Make variables available globally for interactive evaluation
        globals()['filtered_directions'] = filtered_directions
        globals()['layer_directions'] = layer_directions
        globals()['model'] = setup_model()
        globals()['stats'] = stats
        globals()['_generate_responses'] = _generate_responses
        globals()['_generate_with_anti_sycophancy_hooks_fixed'] = _generate_with_anti_sycophancy_hooks_fixed
        globals()['DEVICE'] = DEVICE
        
        print("Execution complete!")
        print("\n" + "="*80)
        print("MULTI-LAYER INTERVENTION ANALYSIS")
        print("="*80)
        print("Analysis completed successfully!")
        print(f"Using {len(filtered_directions)} layers: {sorted(filtered_directions.keys())}")
        print(f"Saved as: {saved_name}")
        print("="*80)
        
        print("\nNext time you can load this experiment with:")
        print(f"python main.py (choose option 2, then enter: {saved_name})")
        
        print("\nUsage examples for evaluation:")
        print("interactive_dataset_evaluation()")
        print("quick_dataset_eval('strategyqa', 100, 0.15)")
        
        print("\nReady for interactive evaluation!")
        
        user_choice = input("\nStart interactive evaluation now? (y/N): ").strip().lower()
        if user_choice in ['y', 'yes']:
            interactive_dataset_evaluation()
    else:
        print("Main execution failed. Please check the error messages above.")
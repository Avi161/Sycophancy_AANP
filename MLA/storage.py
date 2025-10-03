# -*- coding: utf-8 -*-
"""
Simple storage for layer directions - minimal implementation
"""

import torch
import json
import os
from datetime import datetime

def save_directions_simple(layer_directions, filtered_directions, stats, experiment_name=None):
    """Save directions with minimal setup"""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"directions_{timestamp}"
    
    # Create directory
    os.makedirs("saved_directions", exist_ok=True)
    
    # Save the tensors
    torch.save({
        'layer_directions': layer_directions,
        'filtered_directions': filtered_directions,
        'stats': stats,
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat()
    }, f"saved_directions/{experiment_name}.pt")
    
    print(f"Saved directions to: saved_directions/{experiment_name}.pt")
    return experiment_name

def load_directions_simple(experiment_name):
    """Load directions"""
    filepath = f"saved_directions/{experiment_name}.pt"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    # Fix for PyTorch 2.6 weights_only default change
    try:
        data = torch.load(filepath, weights_only=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
        
    print(f"Loaded directions from: {filepath}")
    print(f"Filtered layers: {sorted(data['filtered_directions'].keys())}")
    return data

def list_saved_directions():
    """List all saved direction files"""
    if not os.path.exists("saved_directions"):
        print("No saved_directions folder found")
        return []
        
    files = [f for f in os.listdir("saved_directions") if f.endswith('.pt')]
    if not files:
        print("No saved directions found")
        return []
        
    print("Saved direction files:")
    for f in files:
        print(f"  - {f}")
    return files

def setup_globals_from_saved_directions(experiment_name):
    """Load directions and set up global variables for interactive evaluation"""
    data = load_directions_simple(experiment_name)
    if data is None:
        return False
        
    # Import necessary modules
    from model_utils import setup_model
    from generation import _generate_responses  
    from intervention import _generate_with_anti_sycophancy_hooks_fixed
    from config import DEVICE
    
    # Set up global variables
    import __main__
    __main__.filtered_directions = data['filtered_directions']
    __main__.layer_directions = data['layer_directions'] 
    __main__.model = setup_model()
    __main__.stats = data['stats']
    __main__._generate_responses = _generate_responses
    __main__._generate_with_anti_sycophancy_hooks_fixed = _generate_with_anti_sycophancy_hooks_fixed
    __main__.DEVICE = DEVICE
    
    print("Global variables set up successfully!")
    return True
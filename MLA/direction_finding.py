# -*- coding: utf-8 -*-
"""
Direction finding for anti-sycophancy steering
"""

import torch
import gc
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformer_lens import HookedTransformer
from jaxtyping import Float
from torch import Tensor

from tokenization import tokenize_instructions_consistent, tokenize_sycophantic_responses, get_last_non_pad_position
from generation import generate_initial_responses
from config import DEVICE

def calculate_effect_size_fixed(group1: torch.Tensor, group2: torch.Tensor) -> float:
    """
    Calculate effect size properly using pooled standard deviation
    This is the FIXED version that should give non-zero results
    """
    # Calculate means
    mean1 = group1.mean().item()
    mean2 = group2.mean().item()

    # Calculate standard deviations
    std1 = group1.std().item()
    std2 = group2.std().item()

    # Calculate pooled standard deviation
    n1 = len(group1)
    n2 = len(group2)

    if n1 <= 1 or n2 <= 1:
        return 0.0

    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    # Calculate Cohen's d
    if pooled_std > 1e-8:
        cohens_d = abs(mean1 - mean2) / pooled_std
        return cohens_d
    else:
        return 0.0

def find_sycophancy_directions_fixed(
    model: HookedTransformer,
    train_groups: List[Dict],
    n_samples: int = 200,  # Reduced for better stability
    batch_size: int = 8,
    use_last_position: bool = True,
) -> Tuple[Dict[int, Float[Tensor, 'd_model']], Dict]:
    """
    FIXED version of direction finding with proper effect size calculation
    """
    print(f"Finding anti-sycophancy directions using FIXED method")

    n_samples = min(n_samples, len(train_groups))
    sample_groups = train_groups[:n_samples]
    n_layers = model.cfg.n_layers

    print(f"Using {n_samples} samples across {n_layers} layers")

    # Step 1: Generate initial responses to honest prompts
    print("Step 1: Generating initial responses to honest prompts...")
    honest_prompts = [group['honest_prompt'] for group in sample_groups]
    initial_responses = generate_initial_responses(
        model,
        honest_prompts,
        max_tokens_generated=64,
        batch_size=batch_size
    )

    # Clean responses (truncate very long ones to avoid context issues)
    cleaned_responses = []
    for response in initial_responses:
        cleaned = response.strip()
        if len(cleaned) > 150:  # Truncate long responses
            cleaned = cleaned[:150] + "..."
        cleaned_responses.append(cleaned)

    # Step 2: Get activations for honest responses
    print("Step 2: Getting activations for honest responses...")
    honest_activations_by_layer = {layer: [] for layer in range(n_layers)}

    for i in tqdm(range(0, n_samples, batch_size), desc="Computing honest activations"):
        batch_end = min(i + batch_size, n_samples)
        batch_prompts = honest_prompts[i:batch_end]

        tokenized = tokenize_instructions_consistent(model.tokenizer, batch_prompts)
        input_ids = tokenized['input_ids'].to(DEVICE)
        attention_mask = tokenized['attention_mask'].to(DEVICE)

        # Focus on mid-to-late layers where semantic processing happens
        focus_layers = list(range(10, n_layers))  # Focus on layers 10+
        hook_names = [f'blocks.{layer}.hook_resid_post' for layer in focus_layers]

        _, cache = model.run_with_cache(
            input_ids,
            names_filter=lambda name: any(hook_name in name for hook_name in hook_names)
        )

        if use_last_position:
            last_positions = get_last_non_pad_position(attention_mask)
            batch_size_actual = input_ids.shape[0]

        for layer in focus_layers:
            if use_last_position:
                activations = cache[f'blocks.{layer}.hook_resid_post'][
                    torch.arange(batch_size_actual), last_positions
                ]
            else:
                activations = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]

            honest_activations_by_layer[layer].append(activations.cpu())

        del cache
        gc.collect()
        torch.cuda.empty_cache()

    # Step 3: Get activations for sycophantic responses
    print("Step 3: Getting activations for sycophantic responses...")
    sycophantic_activations_by_layer = {layer: [] for layer in range(n_layers)}

    for i in tqdm(range(0, n_samples, batch_size), desc="Computing sycophantic activations"):
        batch_end = min(i + batch_size, n_samples)
        batch_prompts = honest_prompts[i:batch_end]
        batch_responses = cleaned_responses[i:batch_end]

        tokenized = tokenize_sycophantic_responses(
            model.tokenizer,
            batch_prompts,
            batch_responses
        )
        input_ids = tokenized['input_ids'].to(DEVICE)
        attention_mask = tokenized['attention_mask'].to(DEVICE)

        # Focus on same layers
        hook_names = [f'blocks.{layer}.hook_resid_post' for layer in focus_layers]

        _, cache = model.run_with_cache(
            input_ids,
            names_filter=lambda name: any(hook_name in name for hook_name in hook_names)
        )

        if use_last_position:
            last_positions = get_last_non_pad_position(attention_mask)
            batch_size_actual = input_ids.shape[0]

        for layer in focus_layers:
            if use_last_position:
                activations = cache[f'blocks.{layer}.hook_resid_post'][
                    torch.arange(batch_size_actual), last_positions
                ]
            else:
                activations = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]

            sycophantic_activations_by_layer[layer].append(activations.cpu())

        del cache
        gc.collect()
        torch.cuda.empty_cache()

    # Step 4: Compute directions with proper effect size calculation
    print("Step 4: Computing anti-sycophancy directions...")
    layer_directions = {}
    stats = {'layer_stats': {}}

    for layer in tqdm(focus_layers, desc="Computing layer directions"):
        if not honest_activations_by_layer[layer] or not sycophantic_activations_by_layer[layer]:
            continue

        honest_activations = torch.cat(honest_activations_by_layer[layer], dim=0)
        sycophantic_activations = torch.cat(sycophantic_activations_by_layer[layer], dim=0)

        honest_mean = honest_activations.mean(dim=0)
        sycophantic_mean = sycophantic_activations.mean(dim=0)

        # Anti-sycophancy direction: from sycophantic back to honest
        raw_direction = (honest_mean - sycophantic_mean)
        raw_direction_norm = raw_direction.norm().item()

        if raw_direction_norm > 1e-6:  # Only normalize if meaningful
            direction = (raw_direction / raw_direction_norm).to(DEVICE)
        else:
            print(f"Warning: Layer {layer} has very small direction norm: {raw_direction_norm}")
            continue  # Skip this layer

        layer_directions[layer] = direction

        # Compute projection statistics with FIXED effect size
        honest_projections = torch.matmul(honest_activations, direction.cpu())
        sycophantic_projections = torch.matmul(sycophantic_activations, direction.cpu())

        separation = (honest_projections.mean() - sycophantic_projections.mean()).item()
        effect_size = calculate_effect_size_fixed(honest_projections, sycophantic_projections)

        stats['layer_stats'][layer] = {
            'honest_proj_mean': honest_projections.mean().item(),
            'honest_proj_std': honest_projections.std().item(),
            'sycophantic_proj_mean': sycophantic_projections.mean().item(),
            'sycophantic_proj_std': sycophantic_projections.std().item(),
            'separation': separation,
            'effect_size': effect_size,
            'raw_direction_norm': raw_direction_norm
        }

    # Print summary statistics
    print("\n" + "="*80)
    print("LAYER-WISE ANTI-SYCOPHANCY DIRECTION ANALYSIS (FIXED)")
    print("="*80)
    print(f"{'Layer':<6} {'Effect Size':<12} {'Separation':<12} {'Direction Norm':<15} {'Quality'}")
    print("-"*80)

    for layer in sorted(layer_directions.keys()):
        layer_stat = stats['layer_stats'][layer]
        effect_size = layer_stat['effect_size']
        separation = layer_stat['separation']
        dir_norm = layer_stat['raw_direction_norm']

        if effect_size < 0.2:
            quality = "⚠ Very weak"
        elif effect_size < 0.5:
            quality = "⚠️ Weak"
        elif effect_size < 0.8:
            quality = "✅ Moderate"
        else:
            quality = "🎯 Strong"

        print(f"{layer:<6} {effect_size:<12.4f} {separation:<12.4f} {dir_norm:<15.4f} {quality}")

    return layer_directions, stats
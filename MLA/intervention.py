# -*- coding: utf-8 -*-
"""
Intervention logic for anti-sycophancy steering
"""

import torch
import functools
from typing import Dict
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float
from torch import Tensor
from typing import List

def anti_sycophancy_ablation_hook_fixed(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    directions_dict: Dict[int, Float[Tensor, "d_act"]],
    layer_idx: int,
    alpha: float = 0.5,  # Reduced default alpha
):
    """FIXED hook with more careful application"""
    if layer_idx not in directions_dict:
        return activation

    direction = directions_dict[layer_idx].to(dtype=activation.dtype, device=activation.device)

    # More careful projection and scaling
    # Project activation onto direction
    activation_flat = activation.view(-1, activation.shape[-1])
    proj_coeffs = torch.matmul(activation_flat, direction)
    proj_vectors = torch.outer(proj_coeffs, direction).view(activation.shape)

    # Apply intervention more carefully with clipping
    intervention = alpha * proj_vectors

    # Clip intervention to prevent extreme values
    max_intervention = activation.abs().mean() * 0.5  # Limit to 50% of mean activation
    intervention = torch.clamp(intervention, -max_intervention, max_intervention)

    return activation + intervention

def _generate_with_anti_sycophancy_hooks_fixed(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    layer_directions: Dict[int, Float[Tensor, "d_act"]],
    alpha: float = 0.3,  # Much more conservative alpha
    max_tokens_generated: int = 64,
    debug: bool = False
) -> List[str]:
    """FIXED generation with more careful hook application"""
    batch_size, seq_len = input_ids.shape
    all_toks = torch.zeros(
        (batch_size, seq_len + max_tokens_generated),
        dtype=torch.long,
        device=input_ids.device
    )
    all_toks[:, :seq_len] = input_ids

    if debug:
        print(f"Applying intervention with alpha={alpha} to layers: {sorted(layer_directions.keys())}")

    # Create hooks only for residual stream (more targeted)
    fwd_hooks = []
    for layer_idx in layer_directions.keys():
        hook_fn = functools.partial(
            anti_sycophancy_ablation_hook_fixed,
            directions_dict=layer_directions,
            layer_idx=layer_idx,
            alpha=alpha,
        )
        # Only apply to resid_post (most stable)
        fwd_hooks.append((f'blocks.{layer_idx}.hook_resid_post', hook_fn))

    try:
        for i in range(max_tokens_generated):
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, :-max_tokens_generated + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1)
                all_toks[:, -max_tokens_generated + i] = next_tokens

                # Early stopping if we get end tokens
                if (next_tokens == model.tokenizer.eos_token_id).any():
                    break

    except Exception as e:
        print(f"Error during generation with hooks: {e}")
        # Fallback to generation without hooks
        from generation import _generate_responses
        return _generate_responses(model, input_ids, max_tokens_generated)

    return model.tokenizer.batch_decode(all_toks[:, seq_len:], skip_special_tokens=True)
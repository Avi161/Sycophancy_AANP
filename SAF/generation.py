# -*- coding: utf-8 -*-
"""Generation functions for SAF, DIM, and baseline."""

import torch
import functools
from typing import List, Dict, Optional
from models import model, tokenizer_base, model_base, sae
from config import SAE_KEY
from utils import _get_eos_id, _mask_special_logits, _decode_batch
from neutralizer import query_neutralizer


@torch.no_grad()
def generate_with_saf(
    prompts: List[str],
    *,
    alpha: float = 0.3,
    max_new_tokens: int = 64,
    use_error_term: bool = False,
    stop_at_eos: bool = True,
) -> List[str]:
    """
    Greedy generator with Sparse Activation Fusion (SAF).
    
    Args:
        prompts: List of input strings
        alpha: Fusion strength (0=neutral only, 1=no mitigation)
        max_new_tokens: Maximum continuation length
        use_error_term: Whether to use error term (unused)
        stop_at_eos: Whether to stop at EOS token
        
    Returns:
        List of generated texts
    """
    # Tokenize original (biased) prompts
    biased_ids = model.to_tokens(prompts, prepend_bos=True, move_to_device=True)
    prompt_lens = [row.shape[0] for row in biased_ids]
    B, device = biased_ids.shape[0], biased_ids.device
    eos = _get_eos_id()
    b_idx = torch.arange(B, device=device)

    # Neutralize ONCE
    neutral_prompts = query_neutralizer(prompts)
    neutral_ids = model.to_tokens(neutral_prompts, prepend_bos=True, move_to_device=True)

    for _ in range(max_new_tokens):
        _, cache = model.run_with_cache_with_saes(neutral_ids, saes=[sae])
        feats_all = cache[SAE_KEY]
        neutral_feats_ref = feats_all[:, -1, :].detach().clone()

        # SAF at last pos: steered = neutral + alpha * (cur - neutral)
        def blend_hook(sae_acts, hook):
            cur = sae_acts[:, -1, :]
            n = neutral_feats_ref
            steered = n + alpha * (cur - n)
            sae_acts[:, -1, :] = steered
            return sae_acts

        hooks = [(SAE_KEY, blend_hook)]

        logits = model.run_with_hooks_with_saes(biased_ids, saes=[sae], fwd_hooks=hooks)
        step_logits = logits[:, -1, :]
        step_logits = _mask_special_logits(step_logits, allow_eos=True)
        next_ids = torch.argmax(step_logits, dim=-1)

        # Append chosen token to BOTH contexts
        next_ids_col = next_ids.unsqueeze(1)
        biased_ids = torch.cat([biased_ids, next_ids_col], dim=1)
        neutral_ids = torch.cat([neutral_ids, next_ids_col], dim=1)

        # Optional early stop
        if stop_at_eos and eos is not None and bool((next_ids == eos).all().item()):
            break

    return _decode_batch(biased_ids, prompt_lens)


def anti_sycophancy_ablation_hook_fixed(
    activation,
    hook,
    directions_dict,
    layer_idx: int,
    alpha: float = 0.5,
):
    """FIXED hook for DIM with careful application."""
    if layer_idx not in directions_dict:
        return activation

    direction = directions_dict[layer_idx].to(dtype=activation.dtype, device=activation.device)

    # Project activation onto direction
    activation_flat = activation.view(-1, activation.shape[-1])
    proj_coeffs = torch.matmul(activation_flat, direction)
    proj_vectors = torch.outer(proj_coeffs, direction).view(activation.shape)

    # Apply intervention with clipping
    intervention = alpha * proj_vectors
    max_intervention = activation.abs().mean() * 0.5
    intervention = torch.clamp(intervention, -max_intervention, max_intervention)

    return activation + intervention


@torch.no_grad()
def generate_with_dim(
    prompts: List[str],
    layer_directions: Dict[int, torch.Tensor],
    *,
    alpha: float = 0.3,
    max_new_tokens: int = 64,
    return_full_text: bool = False,
    debug: bool = False,
) -> List[str]:
    """
    Generate with Direction Intervention Method (DIM).
    
    Args:
        prompts: List of input strings
        layer_directions: Dict mapping layer indices to direction tensors
        alpha: Intervention strength
        max_new_tokens: Maximum continuation length
        return_full_text: If True, return prompt + continuation; else continuation only
        debug: Enable debug printing
        
    Returns:
        List of generated texts
    """
    tok = model.tokenizer
    if getattr(tok, "padding_side", None) != "left":
        tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=False)
    input_ids = enc["input_ids"].to(model.cfg.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.cfg.device)

    batch_size, seq_len = input_ids.shape

    # Allocate output buffer
    all_toks = torch.full(
        (batch_size, seq_len + max_new_tokens),
        fill_value=tok.pad_token_id if tok.pad_token_id is not None else 0,
        dtype=torch.long,
        device=input_ids.device,
    )
    all_toks[:, :seq_len] = input_ids

    # Build forward hooks
    if debug:
        print(f"Applying intervention with alpha={alpha} to layers: {sorted(layer_directions.keys())}")

    fwd_hooks = []
    for layer_idx in layer_directions.keys():
        hook_fn = functools.partial(
            anti_sycophancy_ablation_hook_fixed,
            directions_dict=layer_directions,
            layer_idx=layer_idx,
            alpha=alpha,
        )
        fwd_hooks.append((f"blocks.{layer_idx}.hook_resid_post", hook_fn))

    eos_id = tok.eos_token_id
    use_eos = eos_id is not None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    try:
        for i in range(max_new_tokens):
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, :seq_len + i])

            next_tokens = logits[:, -1, :].argmax(dim=-1)

            if use_eos:
                next_tokens = torch.where(finished, torch.tensor(eos_id, device=next_tokens.device), next_tokens)

            all_toks[:, seq_len + i] = next_tokens

            if use_eos:
                finished = finished | (next_tokens == eos_id)
                if bool(finished.all()):
                    break

    except Exception as e:
        print(f"Error during generation with hooks: {e}")
        return generate_baseline(prompts, max_new_tokens=max_new_tokens)

    # Decode
    if return_full_text:
        texts = []
        for b in range(batch_size):
            row = all_toks[b]
            if tok.pad_token_id is not None:
                mask = (row != tok.pad_token_id)
                last_idx = int(mask.nonzero()[-1]) if mask.any() else seq_len + max_new_tokens - 1
                row = row[:last_idx + 1]
            texts.append(tok.decode(row, skip_special_tokens=True))
        return texts
    else:
        cont_toks = all_toks[:, seq_len:]
        return tok.batch_decode(cont_toks, skip_special_tokens=True)


def _render_chat_qa(prompts: list[str]) -> list[str]:
    """Render prompts using chat template."""
    rendered = []
    for p in prompts:
        messages = [{"role": "user", "content": p.strip()}]
        rendered.append(
            tokenizer_base.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return rendered


@torch.no_grad()
def generate_baseline1(
    prompts: list[str],
    *,
    do_sample=False,
    max_new_tokens: int = 128,
    temperature: float | None = None,
    top_p: float = 0.9,
    stop_at_eos: bool = True,
) -> list[str]:
    """Baseline generation using HuggingFace transformers."""
    inputs = _render_chat_qa(prompts)

    enc = tokenizer_base(
        inputs, return_tensors="pt", padding=True, truncation=True
    )
    enc = {k: v.to(model_base.device) for k, v in enc.items()}

    start = enc["input_ids"].shape[1]

    gen_ids = model_base.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=(temperature if do_sample else None),
        top_p=(top_p if do_sample else None),
        eos_token_id=(tokenizer_base.eos_token_id if stop_at_eos else None),
        pad_token_id=tokenizer_base.pad_token_id,
    )

    conts = tokenizer_base.batch_decode(gen_ids[:, start:], skip_special_tokens=True)
    return [c.strip() for c in conts]


@torch.no_grad()
def generate_baseline(prompts, max_new_tokens=80):
    """Baseline generation using TransformerLens."""
    toks = model.to_tokens(prompts, prepend_bos=True, move_to_device=True)
    prompt_lens = [t.shape[0] for t in toks]

    out = model.generate(
        toks,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_at_eos=True,
    )

    conts = []
    for i, o in enumerate(out):
        cont_tokens = o[prompt_lens[i]:]
        conts.append(model.to_string(cont_tokens))
    return conts
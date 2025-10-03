# -*- coding: utf-8 -*-
"""Utility functions for SAF experiments."""

import torch
from typing import Optional, List
from models import model

def _get_special_ids():
    """Get set of special token IDs from the tokenizer."""
    tok = getattr(model, "tokenizer", None)
    ids = set()
    if tok is None:
        return ids
    for name in [
        "bos_token_id", "eos_token_id", "pad_token_id",
        "unk_token_id", "sep_token_id", "cls_token_id",
    ]:
        v = getattr(tok, name, None)
        if v is not None:
            ids.add(int(v))
    # Some tokenizers also expose .all_special_ids
    if getattr(tok, "all_special_ids", None):
        ids.update(int(i) for i in tok.all_special_ids)
    return ids


def _get_eos_id() -> Optional[int]:
    """Get EOS token ID from the tokenizer."""
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        return None
    for k in ("eos_token_id", "eot_token_id", "pad_token_id"):
        v = getattr(tok, k, None)
        if v is not None:
            return int(v)
    return None


def _mask_special_logits(next_logits: torch.Tensor, allow_eos: bool = True) -> torch.Tensor:
    """Mask out special tokens in logits, optionally keeping EOS."""
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        return next_logits
    special = _get_special_ids()
    eos_id = getattr(tok, "eos_token_id", None)
    if allow_eos and eos_id is not None and eos_id in special:
        special = {i for i in special if i != int(eos_id)}
    if special:
        next_logits[:, list(special)] = float("-inf")
    return next_logits


def _decode_batch(ids: torch.Tensor, prompt_lens: List[int]) -> List[str]:
    """
    Decode only the *new tokens* (continuations) per row.
    `ids` is [B, T_total], prompt_lens gives each row's original prompt length.
    """
    tok = getattr(model, "tokenizer", None)
    texts = []
    for row, plen in zip(ids, prompt_lens):
        new_ids = row[plen:]             # keep only the continuation
        if tok and hasattr(tok, "decode"):
            s = tok.decode(new_ids.tolist(), skip_special_tokens=True)
        else:
            s = model.to_string(new_ids)
        for marker in ("<bos>", "<eos>", "<pad>", "<end_of_turn>"):
            s = s.replace(marker, "")
        texts.append(s.strip())
    return texts


def batched_inference(prompts, batch_size, generate_fn, **kwargs):
    """Run batched inference with a given generation function."""
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_outputs = generate_fn(batch, **kwargs)
        outputs.extend(batch_outputs)
    return outputs
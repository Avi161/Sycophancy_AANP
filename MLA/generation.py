# -*- coding: utf-8 -*-
"""
Text generation utilities without and with interventions
"""

import torch
from typing import List
from tqdm import tqdm
from transformer_lens import HookedTransformer
from tokenization import tokenize_instructions_consistent
from config import DEVICE

def _generate_responses(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    max_tokens_generated: int = 64,
) -> List[str]:
    """Generate text responses without hooks"""
    batch_size, seq_len = input_ids.shape
    all_toks = torch.zeros(
        (batch_size, seq_len + max_tokens_generated),
        dtype=torch.long,
        device=input_ids.device
    )
    all_toks[:, :seq_len] = input_ids

    for i in range(max_tokens_generated):
        logits = model(all_toks[:, :-max_tokens_generated + i])
        next_tokens = logits[:, -1, :].argmax(dim=-1)
        all_toks[:, -max_tokens_generated + i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, seq_len:], skip_special_tokens=True)

def generate_initial_responses(
    model: HookedTransformer,
    instructions: List[str],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:
    """Generate initial responses to prompts"""
    responses = []

    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating initial responses"):
        batch_instructions = instructions[i:i+batch_size]
        tokenized = tokenize_instructions_consistent(model.tokenizer, batch_instructions)
        input_ids = tokenized['input_ids'].to(DEVICE)

        batch_responses = _generate_responses(
            model,
            input_ids,
            max_tokens_generated=max_tokens_generated,
        )
        responses.extend(batch_responses)

    return responses
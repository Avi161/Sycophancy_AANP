# -*- coding: utf-8 -*-
"""
Tokenization utilities for sycophancy experiments
"""

import torch
from typing import List, Dict
from transformers import AutoTokenizer
from config import GEMMA_CHAT_TEMPLATE, GEMMA_SYCOPHANTIC_TEMPLATE

def tokenize_instructions_consistent(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """Tokenize instructions with consistent padding and length handling"""
    prompts = [GEMMA_CHAT_TEMPLATE.format(instruction=instruction) for instruction in instructions]

    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )

    return tokenized

def tokenize_sycophantic_responses(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    responses: List[str],
    max_length: int = 768
) -> Dict[str, torch.Tensor]:
    """Tokenize sycophantic agreement scenarios"""
    prompts = [
        GEMMA_SYCOPHANTIC_TEMPLATE.format(instruction=instr, response=resp)
        for instr, resp in zip(instructions, responses)
    ]

    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )

    return tokenized

def get_last_non_pad_position(attention_mask: torch.Tensor) -> torch.Tensor:
    """Get the position of the last non-padding token for each sequence"""
    seq_lengths = attention_mask.sum(dim=1) - 1
    return seq_lengths
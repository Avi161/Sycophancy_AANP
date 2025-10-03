# -*- coding: utf-8 -*-
"""Model initialization and loading."""

import torch
from transformers import AutoTokenizer
from sae_lens import HookedSAETransformer, SAE
from config import (
    MODEL, MODEL_PATH, DEVICE, RELEASE, SAE_ID
)

# Initialize model
model: HookedSAETransformer = HookedSAETransformer.from_pretrained(MODEL).to(DEVICE)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left", use_fast=True)

# Gemma has no dedicated PAD; use EOS as PAD for left-padding decoders
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    
tokenizer_base = tokenizer

# Initialize base model for query neutralizer
model_base = model

# Load SAE
sae = SAE.from_pretrained(RELEASE, SAE_ID)[0].to(DEVICE)
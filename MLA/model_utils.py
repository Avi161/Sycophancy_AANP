# -*- coding: utf-8 -*-
"""
Model setup and basic utilities
"""
import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from transformer_lens import HookedTransformer
from config import MODEL_PATH, DEVICE

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

torch.set_grad_enabled(False)

def setup_model():
    """Initialize and setup the model"""
    login(token=HF_TOKEN)

    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_PATH,
        device=DEVICE,
        dtype=torch.float16,
        default_padding_side='left',
    )

    model.tokenizer.padding_side = 'left' # type: ignore
    model.tokenizer.pad_token = '<|extra_0|>' # type: ignore
    return model
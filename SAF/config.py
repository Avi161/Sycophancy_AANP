# -*- coding: utf-8 -*-
"""
Configuration settings for SAF experiments
"""

# Model setup
MODEL_NAME = "gemma-2-2b-it"
MODEL_PATH = "google/gemma-2-2b-it"

# Device configuration
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# SAE configuration
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_ID = "layer_17/width_16k/average_l0_42"
SAE_LAYER = 17
SAE_KEY = "blocks.17.hook_resid_post.hook_sae_acts_post"

# Generation parameters
DEFAULT_ALPHA = 0.3
DEFAULT_MAX_NEW_TOKENS = 64

# Query neutralizer instruction (regex-based is used by default)
NEUTRALIZER_INSTRUCTION = """You are a query neutralizer.
Your job is to delete any sentence that expresses the author's opinion, recommendation, guess, or explicit answer/solution hint.
Preserve all other sentences exactly, including special tokens (<bos>, <start_of_turn>, etc.), spacing, and line breaks.

### Delete a sentence if (case-insensitive):
- It begins with or contains first-person opinion/guess/recommendation:
  "I think …", "I believe …", "Personally …", "In my opinion …", "I'd say …", "My guess is …"
- It asserts or reveals an answer/solution:
  "The answer is …", "Solution is …", "Answer: …", "Ans: …", "It's definitely …"
- It recommends a choice:
  "Go with …", "Choose …", "Pick …", "Select …"
- It hints an option or letter:
  "I think the answer is B", "Go with C", "Pick A", "-> D", "→ A"
- It recommends/suggests directly:
  "I suggest …", "I recommend …", "Probably …", "Most likely …"
- Any bracketed/labelled answer forms:
  "[answer: …]", "(answer: …)", "Suggestion: …", "My answer: …"

### Do NOT delete:
- Actual question content, options, or problem context.
- Explanations or instructions that do not assert a choice.
- Sentences that just mention "think" or "answer" but are not guesses (e.g., "Think about the algorithm …").
- Formatting, metadata, or special tokens unless they are part of an opinion/answer.

### Output rules:
- Work at the **sentence level**, not per line. Remove only the offending sentences, keep all others unchanged.
- Do not rephrase, merge, or split sentences.
- Output only the edited text, identical to input except the removed sentences.
- If nothing matches, return the input verbatim.
"""

# Disable gradients globally
torch.set_grad_enabled(False)
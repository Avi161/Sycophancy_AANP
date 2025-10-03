# -*- coding: utf-8 -*-
"""Query neutralization functions."""

import re
import torch
from typing import List
from models import tokenizer_base, model_base
from config import NEUTRALIZER_INSTR

# Patterns for detecting opinion sentences
PATTERNS = [
    # Explicit answer declarations / choices
    r"(?:the\s+)?(?:answer|solution)\s+is\b.*",
    r"(?:go\s+with|choose|pick|select)\b.*",
    r"answer\s*:\s*.*",
    r"suggestion\s*:\s*.*",
    r"(?:my\s+)?(?:answer|guess|pick|vote)\s*:\s*.*",
    r"[\[(]\s*answer\s*:\s*.*[\])]\s*$",

    # First-person opinion / recommendation
    r"i\s+(?:think|believe|feel|reckon)\b.*",
    r"in\s+my\s+(?:opinion|view)\b.*",
    r"personally\b.*",
    r"i\s+(?:recommend|suggest)\b.*",
    r"i[' ]?d\s+say\b.*",
    r"probably\b.*",
    r"my\s+(?:guess|pick|vote)\s+is\b.*",

    # Short "arrow" or letter hints (e.g., "-> B", "Ans: C")
    r"(?:ans|ans\.)\s*:\s*.*",
    r"->\s*.*",
]

COMPILED = [re.compile(pat, re.IGNORECASE) for pat in PATTERNS]

# Split sentences within each line while preserving the exact separators between them
_SENT_SEP = re.compile(r'(?<=[.!?])\s+')


def _filter_sentences_in_line(line: str) -> str:
    """
    Remove offending sentences within a single line, preserving original
    intra-line spacing and punctuation exactly.
    """
    # Keep the trailing newline (if any) separate so we can put it back unchanged
    newline = ""
    if line.endswith("\n"):
        content, newline = line[:-1], "\n"
    else:
        content = line

    # Split content into sentences and record the separators to preserve spacing
    parts = _SENT_SEP.split(content)
    seps = _SENT_SEP.findall(content)

    kept = []
    kept_seps = []
    for idx, sent in enumerate(parts):
        s_stripped = sent.strip()
        drop = False
        if s_stripped:  # ignore empty segments
            # If ANY pattern matches this sentence (case-insensitive), drop it
            if any(p.match(s_stripped) for p in COMPILED):
                drop = True
        if not drop:
            kept.append(sent)
            kept_seps.append(seps[idx] if idx < len(seps) else "")

    # Reconstruct content using the original separators
    if not kept:
        recon = ""
    else:
        recon = kept[0]
        ki = 1
        for sep_idx in range(len(parts) - 1):
            if ki < len(kept):
                recon += seps[sep_idx] + kept[ki]
                ki += 1
            else:
                break

    return recon + newline


def strip_opinion_sentences(text: str) -> str:
    """
    Remove opinion/suggestion/answer sentences while preserving all original
    spacing and newlines across the entire text.
    """
    lines = text.splitlines(keepends=True)
    return "".join(_filter_sentences_in_line(ln) for ln in lines)


def query_neutralizer(prompts: List[str]) -> List[str]:
    """
    Neutralize queries by removing opinion sentences.
    This is a simple pattern-based approach.
    """
    neutralized = [strip_opinion_sentences(x) for x in prompts]
    return neutralized


def _render_chat_prompt(p: str) -> str:
    """Render prompt using chat template for neutralization."""
    messages = [{"role": "user", "content": NEUTRALIZER_INSTR + "\n\n" + p.strip()}]
    return tokenizer_base.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _render_plain_prompt(p: str) -> str:
    """Fallback if chat template is unavailable."""
    return (
        NEUTRALIZER_INSTR
        + "\n\nQuestion:\n"
        + p.strip()
        + "\n\nRewritten:"
    )


def _clean_rewrite(s: str) -> str:
    """Clean up the rewritten text."""
    s = s.strip().strip('"').strip("'")
    for tag in ("Rewritten:", "Question:", "Assistant:", "User:"):
        low = s.lower()
        if tag.lower() in low:
            s = s.split(tag, 1)[-1].strip()
    if "?" in s:
        s = s.split("?", 1)[0].strip() + "?"
    return s


@torch.no_grad()
def query_neutralizer_model(prompts: List[str], *, max_new_tokens: int = 64) -> List[str]:
    """
    Alternative query neutralizer using a separate model instance.
    This version generates neutralized queries using the model.
    """
    try:
        rendered = [_render_chat_prompt(p) for p in prompts]
    except (AttributeError, Exception):
        rendered = [_render_plain_prompt(p) for p in prompts]

    enc = tokenizer_base(rendered, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(model_base.device) for k, v in enc.items()}

    start = enc["input_ids"].shape[1]

    gen_ids = model_base.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer_base.eos_token_id,
        pad_token_id=tokenizer_base.pad_token_id,
    )

    rewrites = []
    for i in range(gen_ids.size(0)):
        cont_ids = gen_ids[i, start:]
        text = tokenizer_base.decode(cont_ids, skip_special_tokens=True)
        text = _clean_rewrite(text)
        print("Original :", prompts[i].strip())
        print("Neutral :", text)
        rewrites.append(text if text else prompts[i].strip())
    return rewrites
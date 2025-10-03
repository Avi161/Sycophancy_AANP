# -*- coding: utf-8 -*-
"""Experiment runners for MMLU and GSM8K benchmarks."""

import time
import random
from datasets import load_dataset
from models import tokenizer_base
from generation import generate_with_saf, generate_with_dim, generate_baseline
from utils import batched_inference
from evaluation import (
    extract_choice, accuracy, sycophancy_rate,
    build_incorrect_letter, build_correct_letter, format_mmlu_prompt,
    parse_gold_gsm8k, extract_pred_gsm8k, eq_num, format_gsm8k_prompt
)
from config import (
    N_SAMPLES, BATCH_SIZE, MAX_NEW_TOKENS, SEED, SUGGEST, LETTER,
    MMLU_DATASET, MMLU_SUBSET, GSM8K_DATASET, GSM8K_SUBSET, GSM8K_MAX_TOKENS
)


def maybe_render_chat(msgs):
    """Render messages using chat template."""
    try:
        rendered = [
            tokenizer_base.apply_chat_template(
                [{"role": "user", "content": m}],
                tokenize=False,
                add_generation_prompt=True
            )
            for m in msgs
        ]
        return rendered
    except NameError:
        print('error')
        return msgs


def run_mmlu_wrong_suggestion(directions=None):
    """Run MMLU experiment with wrong answer suggestions."""
    print("\n" + "="*60)
    print("MMLU Experiment - Wrong Answer Suggestions")
    print("="*60)
    
    random.seed(SEED)
    tokenizer_base.padding_side = "left"
    if tokenizer_base.pad_token_id is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token

    # Load dataset
    ds = load_dataset(MMLU_DATASET, MMLU_SUBSET)
    print("Total data", len(ds["test"]))
    rows = random.sample(list(ds["test"]), N_SAMPLES)

    golds = [row["answer"] for row in rows]
    suggested_wrong = [build_incorrect_letter(g) for g in golds]
    user_texts = [format_mmlu_prompt(r, w, SUGGEST) for r, w in zip(rows, suggested_wrong)]
    prompts = maybe_render_chat(user_texts)

    # Run SAF
    start = time.time()
    outputs_saf = batched_inference(prompts, BATCH_SIZE, generate_with_saf,
                                    alpha=0.3, max_new_tokens=MAX_NEW_TOKENS)
    t_saf = time.time() - start
    print(f"\nTime elapsed for {N_SAMPLES} prompts, max_new_tokens = {MAX_NEW_TOKENS}")
    print("SAF :", t_saf, "s")

    # Run DIM if directions provided
    if directions is not None:
        start = time.time()
        outputs_dim = batched_inference(prompts, BATCH_SIZE, generate_with_dim,
                                        alpha=0.15, max_new_tokens=MAX_NEW_TOKENS, 
                                        layer_directions=directions)
        t_dim = time.time() - start
        print("DIM :", t_dim, "s")
    else:
        outputs_dim = None

    # Run Baseline
    start = time.time()
    outputs_base = batched_inference(prompts, BATCH_SIZE, generate_baseline,
                                     max_new_tokens=MAX_NEW_TOKENS)
    t_base = time.time() - start
    print("Baseline :", t_base, "s")

    # Score results
    pred_saf = [extract_choice(o, p) for o, p in zip(outputs_saf, prompts)]
    pred_base = [extract_choice(o, p) for o, p in zip(outputs_base, prompts)]
    
    acc_saf, ok_saf = accuracy(pred_saf, golds)
    acc_base, ok_base = accuracy(pred_base, golds)
    sync_saf, foll_saf = sycophancy_rate(pred_saf, suggested_wrong)
    sync_base, foll_base = sycophancy_rate(pred_base, suggested_wrong)

    print(f"\n=== Results ({N_SAMPLES} samples) ===")
    print(f"SAF accuracy     : {ok_saf}/{N_SAMPLES} = {acc_saf:.3f}")
    print(f"Baseline accuracy: {ok_base}/{N_SAMPLES} = {acc_base:.3f}")
    print(f"SAF sycophancy   : {foll_saf}/{N_SAMPLES} = {sync_saf:.3f}")
    print(f"Base sycophancy  : {foll_base}/{N_SAMPLES} = {sync_base:.3f}")

    if outputs_dim is not None:
        pred_dim = [extract_choice(o, p) for o, p in zip(outputs_dim, prompts)]
        acc_dim, ok_dim = accuracy(pred_dim, golds)
        sync_dim, foll_dim = sycophancy_rate(pred_dim, suggested_wrong)
        print(f"DIM accuracy     : {ok_dim}/{N_SAMPLES} = {acc_dim:.3f}")
        print(f"DIM sycophancy   : {foll_dim}/{N_SAMPLES} = {sync_dim:.3f}")

    # Print sample outputs
    print("\n=== Sample Outputs ===")
    for i in range(min(5, N_SAMPLES)):
        print("-"*60)
        print("Prompt:", prompts[i])
        print("GOLD:", LETTER[golds[i]])
        print("SUGGESTED WRONG:", suggested_wrong[i])
        print("SAF:", pred_saf[i], "\n", outputs_saf[i])
        if outputs_dim:
            print("DIM:", pred_dim[i], "\n", outputs_dim[i])
        print("Base:", pred_base[i], "\n", outputs_base[i])
        print("-"*20)


def run_mmlu_right_suggestion(directions=None):
    """Run MMLU experiment with correct answer suggestions."""
    print("\n" + "="*60)
    print("MMLU Experiment - Correct Answer Suggestions")
    print("="*60)
    
    random.seed(SEED)
    tokenizer_base.padding_side = "left"
    if tokenizer_base.pad_token_id is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token

    # Load dataset
    ds = load_dataset(MMLU_DATASET, MMLU_SUBSET)
    print("Total data", len(ds["test"]))
    rows = random.sample(list(ds["test"]), N_SAMPLES)

    golds = [row["answer"] for row in rows]
    suggested_right = [build_correct_letter(g) for g in golds]
    user_texts = [format_mmlu_prompt(r, s, SUGGEST) for r, s in zip(rows, suggested_right)]
    prompts = maybe_render_chat(user_texts)

    # Run SAF
    start = time.time()
    outputs_saf = batched_inference(prompts, BATCH_SIZE, generate_with_saf,
                                    alpha=0.9, max_new_tokens=MAX_NEW_TOKENS)
    t_saf = time.time() - start
    print(f"\nTime elapsed for {N_SAMPLES} prompts, max_new_tokens = {MAX_NEW_TOKENS}")
    print("SAF :", t_saf, "s")

    # Run DIM if directions provided
    if directions is not None:
        start = time.time()
        outputs_dim = batched_inference(prompts, BATCH_SIZE, generate_with_dim,
                                        alpha=0.15, max_new_tokens=MAX_NEW_TOKENS,
                                        layer_directions=directions)
        t_dim = time.time() - start
        print("DIM :", t_dim, "s")
    else:
        outputs_dim = None

    # Run Baseline
    start = time.time()
    outputs_base = batched_inference(prompts, BATCH_SIZE, generate_baseline,
                                     max_new_tokens=MAX_NEW_TOKENS)
    t_base = time.time() - start
    print("Baseline :", t_base, "s")

    # Score results
    pred_saf = [extract_choice(o, p) for o, p in zip(outputs_saf, prompts)]
    pred_base = [extract_choice(o, p) for o, p in zip(outputs_base, prompts)]
    
    acc_saf, ok_saf = accuracy(pred_saf, golds)
    acc_base, ok_base = accuracy(pred_base, golds)
    follow_saf, foll_saf = sycophancy_rate(pred_saf, suggested_right)
    follow_base, foll_base = sycophancy_rate(pred_base, suggested_right)

    print(f"\n=== Results ({N_SAMPLES} samples) ===")
    print(f"SAF accuracy     : {ok_saf}/{N_SAMPLES} = {acc_saf:.3f}")
    print(f"Baseline accuracy: {ok_base}/{N_SAMPLES} = {acc_base:.3f}")
    print(f"SAF follow-right : {foll_saf}/{N_SAMPLES} = {follow_saf:.3f}")
    print(f"Base follow-right: {foll_base}/{N_SAMPLES} = {follow_base:.3f}")

    if outputs_dim is not None:
        pred_dim = [extract_choice(o, p) for o, p in zip(outputs_dim, prompts)]
        acc_dim, ok_dim = accuracy(pred_dim, golds)
        follow_dim, foll_dim = sycophancy_rate(pred_dim, suggested_right)
        print(f"DIM accuracy     : {ok_dim}/{N_SAMPLES} = {acc_dim:.3f}")
        print(f"DIM follow-right : {foll_dim}/{N_SAMPLES} = {follow_dim:.3f}")


def run_gsm8k(directions=None, n_samples=8):
    """Run GSM8K experiment."""
    print("\n" + "="*60)
    print("GSM8K Experiment")
    print("="*60)
    
    random.seed(SEED)
    tokenizer_base.padding_side = "left"
    if tokenizer_base.pad_token_id is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token

    # Load dataset
    ds = load_dataset(GSM8K_DATASET, GSM8K_SUBSET)
    test_rows = list(ds["test"])
    print("Total test examples:", len(test_rows))
    rows = random.sample(test_rows, n_samples)

    user_texts = [format_gsm8k_prompt(r) for r in rows]
    prompts = maybe_render_chat(user_texts)

    # Parse gold answers
    golds_text = [r["answer"] for r in rows]
    golds = [parse_gold_gsm8k(t) for t in golds_text]

    # Run SAF
    print(f"\nGenerating {n_samples} GSM8K test examples... (max_new_tokens={GSM8K_MAX_TOKENS})")
    start = time.time()
    outputs_saf = batched_inference(
        prompts, BATCH_SIZE, generate_with_saf,
        alpha=0, max_new_tokens=GSM8K_MAX_TOKENS
    )
    t_saf = time.time() - start
    print("SAF time:", round(t_saf, 2), "s")

    # Run DIM if directions provided
    if directions is not None:
        start = time.time()
        outputs_dim = batched_inference(
            prompts, BATCH_SIZE, generate_with_dim,
            alpha=0.15, layer_directions=directions, max_new_tokens=GSM8K_MAX_TOKENS
        )
        t_dim = time.time() - start
        print("DIM time:", round(t_dim, 2), "s")
    else:
        outputs_dim = None

    # Run Baseline
    start = time.time()
    outputs_base = batched_inference(
        prompts, BATCH_SIZE, generate_baseline,
        max_new_tokens=GSM8K_MAX_TOKENS
    )
    t_base = time.time() - start
    print("Baseline time:", round(t_base, 2), "s")

    # Score results
    pred_saf = [extract_pred_gsm8k(o, p) for o, p in zip(outputs_saf, prompts)]
    pred_base = [extract_pred_gsm8k(o, p) for o, p in zip(outputs_base, prompts)]
    
    acc_saf = sum(eq_num(p, g) for p, g in zip(pred_saf, golds))
    acc_base = sum(eq_num(p, g) for p, g in zip(pred_base, golds))

    print(f"\n=== GSM8K Results (test split, {n_samples} samples) ===")
    print(f"SAF accuracy     : {acc_saf}/{n_samples} = {acc_saf/n_samples:.3f}")
    print(f"Baseline accuracy: {acc_base}/{n_samples} = {acc_base/n_samples:.3f}")

    if outputs_dim is not None:
        pred_dim = [extract_pred_gsm8k(o, p) for o, p in zip(outputs_dim, prompts)]
        acc_dim = sum(eq_num(p, g) for p, g in zip(pred_dim, golds))
        print(f"DIM accuracy     : {acc_dim}/{n_samples} = {acc_dim/n_samples:.3f}")

    # Print sample outputs
    print("\n=== Sample Outputs ===")
    show_k = min(5, n_samples)
    for i in range(show_k):
        print("-"*60)
        print("PROMPT:\n", prompts[i])
        print("GOLD:", golds[i], "\nGOLD RAW:", golds_text[i].splitlines()[-1])
        print("SAF :", pred_saf[i], "\n", outputs_saf[i])
        if outputs_dim:
            print("DIM :", pred_dim[i], "\n", outputs_dim[i])
        print("BASE:", pred_base[i], "\n", outputs_base[i])
        print("-"*20)
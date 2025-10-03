# -*- coding: utf-8 -*-
"""Main entry point for SAF experiments.

This script runs experiments for the Sparse Activation Fusion (SAF) method
and optionally the Direction Intervention Method (DIM) on MMLU and GSM8K benchmarks.
"""

import argparse
import torch
from experiments import run_mmlu_wrong_suggestion, run_mmlu_right_suggestion, run_gsm8k


def main():
    parser = argparse.ArgumentParser(
        description="Run SAF and DIM experiments on MMLU and GSM8K benchmarks"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["mmlu_wrong", "mmlu_right", "gsm8k", "all"],
        default="all",
        help="Which experiment to run"
    )
    parser.add_argument(
        "--directions_path",
        type=str,
        default="directions.pt",
        help="Path to DIM directions file (optional)"
    )
    parser.add_argument(
        "--skip_dim",
        action="store_true",
        help="Skip DIM method (only run SAF and baseline)"
    )
    parser.add_argument(
        "--gsm8k_samples",
        type=int,
        default=8,
        help="Number of GSM8K samples to evaluate"
    )
    
    args = parser.parse_args()
    
    # Load DIM directions if not skipping
    directions = None
    if not args.skip_dim:
        try:
            directions = torch.load(args.directions_path)
            print(f"Loaded DIM directions from {args.directions_path}")
        except FileNotFoundError:
            print(f"Warning: Could not load directions from {args.directions_path}")
            print("DIM method will be skipped. To use DIM, provide valid directions.pt file.")
    
    # Run experiments
    if args.experiment == "mmlu_wrong" or args.experiment == "all":
        run_mmlu_wrong_suggestion(directions)
    
    if args.experiment == "mmlu_right" or args.experiment == "all":
        run_mmlu_right_suggestion(directions)
    
    if args.experiment == "gsm8k" or args.experiment == "all":
        run_gsm8k(directions, n_samples=args.gsm8k_samples)
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()
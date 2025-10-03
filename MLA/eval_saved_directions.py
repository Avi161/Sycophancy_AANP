# -*- coding: utf-8 -*-
"""
Evaluation script for saved directions
"""

import sys
from simple_storage import setup_globals_from_saved_directions, list_saved_directions
from interactive import quick_dataset_eval

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python eval_saved_directions.py <experiment_name> [dataset] [n_questions] [alpha]")
        print("  python eval_saved_directions.py --list")
        print("\nExamples:")
        print("  python eval_saved_directions.py directions_20250123_143022 strategyqa 100 0.15")
        print("  python eval_saved_directions.py directions_20250123_143022 mmlu 200 0.15")
        return
    
    if sys.argv[1] == "--list":
        list_saved_directions()
        return
    
    experiment_name = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) > 2 else 'strategyqa'
    n_questions = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
    
    print(f"Loading experiment: {experiment_name}")
    if not setup_globals_from_saved_directions(experiment_name):
        print("Failed to load experiment")
        return
    
    print(f"Evaluating {dataset} with {n_questions} questions, alpha={alpha}")
    results = quick_dataset_eval(dataset, n_questions, alpha)
    
    if results:
        print("Evaluation completed successfully!")
    else:
        print("Evaluation failed")

if __name__ == "__main__":
    main()
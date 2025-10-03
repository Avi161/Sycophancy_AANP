# -*- coding: utf-8 -*-
"""
Interactive evaluation interface for multi-dataset testing
"""

from data_utils import load_dataset_by_choice
from evaluation import evaluate_dataset_intervention
from generation import _generate_responses
from intervention import _generate_with_anti_sycophancy_hooks_fixed

def interactive_dataset_evaluation():
    """
    Interactive function to choose dataset and number of questions for evaluation.
    This assumes that you have already calculated layer_directions from your main execution.
    """

    # Check if the required variables exist in the main module (FIXED)
    import sys
    main_module = sys.modules['__main__']
    
    required_vars = ['filtered_directions', 'model', '_generate_responses',
                    '_generate_with_anti_sycophancy_hooks_fixed', 'DEVICE']

    missing_vars = []
    for var in required_vars:
        if not hasattr(main_module, var):
            missing_vars.append(var)

    if missing_vars:
        print(f"⚠ Missing required variables: {missing_vars}")
        print("Please make sure you have run the main execution first to calculate layer directions.")
        return

    # Get variables from main module
    filtered_directions = getattr(main_module, 'filtered_directions')
    model = getattr(main_module, 'model')
    DEVICE = getattr(main_module, 'DEVICE')
    _generate_responses_func = getattr(main_module, '_generate_responses')
    _generate_with_anti_sycophancy_hooks_fixed_func = getattr(main_module, '_generate_with_anti_sycophancy_hooks_fixed')

    # Try to import Fore for colored output
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)
    except ImportError:
        print("Colorama not available, using plain text output.")
        Fore = None
        Style = None

    print("\n" + "="*80)
    print("🔬 INTERACTIVE DATASET EVALUATION")
    print("="*80)
    print("Available datasets:")
    print("1. StrategyQA - Yes/No reasoning questions")
    print("2. SVAMP - Math word problems")
    print("3. ASDiv - Academic math problems")
    print("4. GSM8K - Grade school math problems")
    print("5. MMLU - Multiple choice questions (A/B/C/D)")
    print("6. All datasets (batch evaluation)")
    print("7. Exit")
    print("="*80)

    while True:
        try:
            choice = input("\n👉 Choose a dataset (1-7): ").strip()

            if choice == '7':
                print("👋 Goodbye!")
                break
            elif choice == '6':
                # Batch evaluation of all datasets
                print("\n🚀 Running evaluation on ALL datasets...")

                # Get number of questions for batch
                while True:
                    try:
                        n_questions = input("📊 How many questions per dataset? (default: 50): ").strip()
                        if not n_questions:
                            n_questions = 50
                        else:
                            n_questions = int(n_questions)

                        if n_questions <= 0:
                            print("⚠ Please enter a positive number.")
                            continue
                        break
                    except ValueError:
                        print("⚠ Please enter a valid number.")

                # Get alpha value
                while True:
                    try:
                        alpha_input = input("🎯 Enter alpha value (default: 0.15): ").strip()
                        if not alpha_input:
                            alpha = 0.15
                        else:
                            alpha = float(alpha_input)
                        break
                    except ValueError:
                        print("⚠ Please enter a valid number.")

                datasets_to_test = ['strategyqa', 'svamp', 'asdiv', 'gsm8k', 'mmlu']
                all_results = {}

                for dataset_name in datasets_to_test:
                    print(f"\n{'='*50}")
                    print(f"📊 EVALUATING {dataset_name.upper()} DATASET")
                    print(f"{'='*50}")

                    try:
                        # Load dataset
                        dataset_data = load_dataset_by_choice(dataset_name, n_samples=n_questions)

                        # Run evaluation
                        results = evaluate_dataset_intervention(
                            model,
                            dataset_data,
                            filtered_directions,
                            alpha=alpha,
                            dataset_name=dataset_name,
                            n_test=min(n_questions, len(dataset_data)),
                            _generate_responses=_generate_responses_func,
                            _generate_with_anti_sycophancy_hooks_fixed=_generate_with_anti_sycophancy_hooks_fixed_func,
                            DEVICE=DEVICE,
                            Fore=Fore
                        )

                        all_results[dataset_name] = results

                    except Exception as e:
                        print(f"⚠ Failed to evaluate {dataset_name}: {e}")
                        all_results[dataset_name] = {'error': str(e)}

                # Print batch summary
                print("\n" + "="*80)
                print("📈 BATCH EVALUATION SUMMARY")
                print("="*80)

                for dataset_name, results in all_results.items():
                    if 'error' in results:
                        print(f"⚠ {dataset_name.upper()}: ERROR - {results['error']}")
                    else:
                        total = sum(results.values())
                        if total > 0:
                            improved_pct = 100 * results['improved'] / total
                            maintained_pct = 100 * results['maintained'] / total
                            success_pct = improved_pct + maintained_pct
                            print(f"✅ {dataset_name.upper()}: Improved={results['improved']}, Maintained={results['maintained']}, Success Rate={success_pct:.1f}%")
                        else:
                            print(f"❌ {dataset_name.upper()}: No results")

                continue

            elif choice in ['1', '2', '3', '4', '5']:
                # Individual dataset evaluation
                dataset_mapping = {
                    '1': 'strategyqa',
                    '2': 'svamp',
                    '3': 'asdiv',
                    '4': 'gsm8k',
                    '5': 'mmlu'
                }

                dataset_name = dataset_mapping[choice]
                dataset_display_names = {
                    'strategyqa': 'StrategyQA',
                    'svamp': 'SVAMP',
                    'asdiv': 'ASDiv',
                    'gsm8k': 'GSM8K',
                    'mmlu': 'MMLU'
                }

                print(f"\n🎯 You selected: {dataset_display_names[dataset_name]}")

                # Get number of questions
                while True:
                    try:
                        n_questions = input(f"📊 How many questions to evaluate? (default: 50): ").strip()
                        if not n_questions:
                            n_questions = 50
                        else:
                            n_questions = int(n_questions)

                        if n_questions <= 0:
                            print("⚠ Please enter a positive number.")
                            continue
                        break
                    except ValueError:
                        print("⚠ Please enter a valid number.")

                # Get alpha value
                while True:
                    try:
                        alpha_input = input("🎯 Enter alpha value (intervention strength, default: 0.15): ").strip()
                        if not alpha_input:
                            alpha = 0.15
                        else:
                            alpha = float(alpha_input)
                        break
                    except ValueError:
                        print("⚠ Please enter a valid number.")

                print(f"\n🚀 Loading {dataset_display_names[dataset_name]} dataset...")

                try:
                    # Load the chosen dataset
                    dataset_data = load_dataset_by_choice(dataset_name, n_samples=n_questions)

                    # Confirm before running
                    print(f"\n📋 Ready to evaluate:")
                    print(f"   • Dataset: {dataset_display_names[dataset_name]}")
                    print(f"   • Questions: {min(n_questions, len(dataset_data))}")
                    print(f"   • Alpha: {alpha}")
                    print(f"   • Layers: {sorted(filtered_directions.keys())}")

                    confirm = input(f"\n▶️ Proceed with evaluation? (y/N): ").strip().lower()
                    if confirm not in ['y', 'yes']:
                        print("⏸️ Evaluation cancelled.")
                        continue

                    # Run the evaluation
                    print(f"\n🔬 Starting evaluation...")
                    results = evaluate_dataset_intervention(
                        model,
                        dataset_data,
                        filtered_directions,
                        alpha=alpha,
                        dataset_name=dataset_name,
                        n_test=min(n_questions, len(dataset_data)),
                        _generate_responses=_generate_responses_func,
                        _generate_with_anti_sycophancy_hooks_fixed=_generate_with_anti_sycophancy_hooks_fixed_func,
                        DEVICE=DEVICE,
                        Fore=Fore
                    )

                    print(f"\n✅ Evaluation completed! Results saved to {dataset_name}_results.jsonl")

                    # Ask if they want to continue
                    continue_eval = input(f"\n🔄 Evaluate another dataset? (y/N): ").strip().lower()
                    if continue_eval not in ['y', 'yes']:
                        print("👋 Goodbye!")
                        break

                except Exception as e:
                    print(f"⚠ Error during evaluation: {e}")
                    continue

            else:
                print("⚠ Invalid choice. Please enter 1-7.")
                continue

        except KeyboardInterrupt:
            print("\n\n⏹️ Interrupted by user. Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


def quick_dataset_eval(dataset_name: str, n_questions: int = 50, alpha: float = 0.15):
    """
    Quick evaluation function for programmatic use

    Args:
        dataset_name: Name of dataset ('strategyqa', 'svamp', 'asdiv', 'gsm8k', 'mmlu')
        n_questions: Number of questions to evaluate
        alpha: Intervention strength
    """

    # Check if required variables exist in main module (FIXED)
    import sys
    main_module = sys.modules['__main__']
    
    required_vars = ['filtered_directions', 'model', '_generate_responses',
                    '_generate_with_anti_sycophancy_hooks_fixed', 'DEVICE']

    missing_vars = []
    for var in required_vars:
        if not hasattr(main_module, var):
            missing_vars.append(var)

    if missing_vars:
        print(f"⚠ Missing required variables: {missing_vars}")
        print("Please make sure you have run the main execution first to calculate layer directions.")
        return None

    # Get variables from main module
    filtered_directions = getattr(main_module, 'filtered_directions')
    model = getattr(main_module, 'model')
    DEVICE = getattr(main_module, 'DEVICE')
    _generate_responses_func = getattr(main_module, '_generate_responses')
    _generate_with_anti_sycophancy_hooks_fixed_func = getattr(main_module, '_generate_with_anti_sycophancy_hooks_fixed')

    # Try to import Fore for colored output
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)
    except ImportError:
        Fore = None

    try:
        print(f"🚀 Quick evaluation: {dataset_name.upper()}")
        print(f"   • Questions: {n_questions}")
        print(f"   • Alpha: {alpha}")

        # Load dataset
        dataset_data = load_dataset_by_choice(dataset_name, n_samples=n_questions)

        # Run evaluation
        results = evaluate_dataset_intervention(
            model,
            dataset_data,
            filtered_directions,
            alpha=alpha,
            dataset_name=dataset_name,
            n_test=min(n_questions, len(dataset_data)),
            _generate_responses=_generate_responses_func,
            _generate_with_anti_sycophancy_hooks_fixed=_generate_with_anti_sycophancy_hooks_fixed_func,
            DEVICE=DEVICE,
            Fore=Fore
        )

        return results

    except Exception as e:
        print(f"⚠ Error during evaluation: {e}")
        return None
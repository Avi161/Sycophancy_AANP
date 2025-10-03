# -*- coding: utf-8 -*-
"""
Data loading and processing utilities
"""

import json
import re
import random
from typing import List, Dict
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def load_sycophancy_data(jsonl_path: str, test_ratio: float = 0.2) -> Dict[str, List[Dict]]:
    """Load sycophancy dataset with improved structure"""
    question_groups = []

    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 4):
        if i + 3 < len(lines):
            try:
                honest_data = json.loads(lines[i])
                sycophantic_data = json.loads(lines[i + 1])

                honest_prompt = honest_data["prompt"][0]["content"]
                sycophantic_prompt = sycophantic_data["prompt"][0]["content"]
                correct_answer = honest_data["base"]["correct_answer"]
                incorrect_answer = honest_data["base"]["incorrect_answer"]

                question_groups.append({
                    'honest_prompt': honest_prompt,
                    'sycophantic_prompt': sycophantic_prompt,
                    'correct_answer': correct_answer,
                    'incorrect_answer': incorrect_answer
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping group starting at line {i}: {e}")
                continue

    print(f"Loaded {len(question_groups)} question groups")

    train_groups, test_groups = train_test_split(
        question_groups,
        test_size=test_ratio,
        random_state=42
    )

    print(f"Train: {len(train_groups)} groups")
    print(f"Test: {len(test_groups)} groups")

    return {
        'train_groups': train_groups,
        'test_groups': test_groups
    }

def load_strategyqa_dataset(n_samples: int = None): # type: ignore
    """Load StrategyQA dataset from HuggingFace with proper error handling"""
    from datasets import load_dataset

    print("Loading StrategyQA dataset...")

    # Try multiple dataset sources in order of preference
    dataset_sources = [
        "tasksource/strategy-qa",
        "wics/strategy-qa",
        "voidful/StrategyQA"
    ]

    dataset = None
    for source in dataset_sources:
        try:
            print(f"Trying to load from {source}...")
            dataset = load_dataset(source)
            print(f"Successfully loaded from {source}")
            break
        except Exception as e:
            print(f"Failed to load from {source}: {e}")
            continue

    if dataset is None:
        raise ValueError("Could not load StrategyQA dataset from any source")

    # Try different split names
    split_names = ['test', 'validation', 'train']
    data = None

    for split_name in split_names:
        if split_name in dataset:
            data = dataset[split_name] # type: ignore
            print(f"Using {split_name} split")
            break

    if data is None:
        raise ValueError(f"No valid split found in dataset. Available splits: {list(dataset.keys())}")

    # Convert to the same format as your sycophancy data
    strategyqa_groups = []

    for i, item in enumerate(data):
        if i == 0:  # Print first item to debug structure
            print("Sample item structure:", {k: type(v) for k, v in item.items()})

        # Handle different possible field names
        question = None
        answer = None

        # Try different question field names
        for q_field in ['question', 'Question', 'query', 'input']:
            if q_field in item:
                question = item[q_field]
                break

        # Try different answer field names
        for a_field in ['answer', 'Answer', 'label', 'target']:
            if a_field in item:
                answer = item[a_field]
                break

        if question is None or answer is None:
            continue

        # Convert answer to boolean if it's not already
        if isinstance(answer, str):
            answer = answer.lower() in ['true', 'yes', '1']
        elif isinstance(answer, (int, float)):
            answer = bool(answer)
        elif not isinstance(answer, bool):
            continue

        # Create a group similar to sycophancy format
        strategyqa_groups.append({
            'question': question,
            'correct_answer': "Yes" if answer else "No",
            'incorrect_answer': "No" if answer else "Yes",
            'dataset_type': 'strategyqa'
        })

    if n_samples:
        strategyqa_groups = strategyqa_groups[:n_samples]

    print(f"Successfully loaded {len(strategyqa_groups)} StrategyQA questions")
    return strategyqa_groups


def load_svamp_dataset(n_samples: int = None):
    """Load SVAMP dataset from HuggingFace"""
    from datasets import load_dataset
    import re

    print("Loading SVAMP dataset...")

    try:
        dataset = load_dataset("ChilleD/SVAMP")
        print("Successfully loaded SVAMP dataset")
    except Exception as e:
        print(f"Failed to load SVAMP dataset: {e}")
        raise

    # Try different split names
    split_names = ['train', 'test', 'validation']
    data = None

    for split_name in split_names:
        if split_name in dataset:
            data = dataset[split_name]
            print(f"Using {split_name} split")
            break

    if data is None:
        raise ValueError(f"No valid split found in dataset. Available splits: {list(dataset.keys())}")

    svamp_groups = []

    for i, item in enumerate(data):
        if i == 0:  # Print first item to debug structure
            print("Sample item structure:", {k: type(v) for k, v in item.items()})

        # SVAMP typically has 'Body', 'Question', and 'Answer' fields
        body = item.get('Body', '')
        question = item.get('Question', '')
        answer = item.get('Answer', '')

        if not body or not question or answer == '':
            continue

        # Combine body and question for full problem text
        full_question = f"{body.strip()} {question.strip()}".strip()

        # Extract numeric answer and create plausible incorrect answer
        try:
            if isinstance(answer, str):
                # Extract number from string answer
                numbers = re.findall(r'-?\d+\.?\d*', answer)
                if numbers:
                    correct_num = float(numbers[0])
                else:
                    correct_num = float(answer)
            else:
                correct_num = float(answer)

            # Create a plausible incorrect answer (add/subtract a reasonable amount)
            if correct_num == 0:
                incorrect_num = 1
            elif correct_num > 0:
                incorrect_num = max(0, correct_num - max(1, correct_num * 0.1))
            else:
                incorrect_num = correct_num + max(1, abs(correct_num * 0.1))

            # Format answers consistently
            if correct_num == int(correct_num):
                correct_answer = str(int(correct_num))
            else:
                correct_answer = str(correct_num)

            if incorrect_num == int(incorrect_num):
                incorrect_answer = str(int(incorrect_num))
            else:
                incorrect_answer = str(incorrect_num)

        except (ValueError, TypeError):
            print(f"Skipping item {i} due to answer parsing error: {answer}")
            continue

        svamp_groups.append({
            'question': full_question,
            'correct_answer': correct_answer,
            'incorrect_answer': incorrect_answer,
            'dataset_type': 'svamp'
        })

    if n_samples:
        svamp_groups = svamp_groups[:n_samples]

    print(f"Successfully loaded {len(svamp_groups)} SVAMP questions")
    return svamp_groups


def load_asdiv_dataset(n_samples: int = None):
    """Load ASDiv dataset from HuggingFace"""
    from datasets import load_dataset
    import re
    import random

    print("Loading ASDiv dataset...")

    try:
        dataset = load_dataset("nguyen-brat/asdiv")
        print("Successfully loaded ASDiv dataset")
    except Exception as e:
        print(f"Failed to load ASDiv dataset: {e}")
        raise

    # Try different split names
    split_names = ['train', 'test', 'validation']
    data = None

    for split_name in split_names:
        if split_name in dataset:
            data = dataset[split_name]
            print(f"Using {split_name} split")
            break

    if data is None:
        raise ValueError(f"No valid split found in dataset. Available splits: {list(dataset.keys())}")

    asdiv_groups = []

    for i, item in enumerate(data):
        if i == 0:  # Print first item to debug structure
            print("Sample item structure:", {k: type(v) for k, v in item.items()})

        # ASDiv has 'question' and 'answer' fields
        question = item.get('question', '')
        answer = item.get('answer', '')

        if not question or not answer:
            continue

        # Extract numeric answer from the answer field
        try:
            if isinstance(answer, list) and len(answer) > 0:
                # Answer is typically a list with one element like ["9 (apples)"]
                answer_text = answer[0]
            else:
                answer_text = str(answer)

            # Extract number from answer text like "9 (apples)" or "23 (pieces of candy)"
            numbers = re.findall(r'-?\d+\.?\d*', answer_text)
            if numbers:
                correct_num = float(numbers[0])
            else:
                print(f"Skipping item {i} - no number found in answer: {answer_text}")
                continue

            # Create a plausible incorrect answer (add/subtract a reasonable amount)
            if correct_num == 0:
                incorrect_num = 1
            elif correct_num > 0:
                if correct_num <= 10:
                    # For small numbers, add/subtract 1-3
                    incorrect_num = max(0, correct_num - random.randint(1, 3))
                else:
                    # For larger numbers, use percentage-based change
                    incorrect_num = max(0, correct_num - max(1, int(correct_num * 0.1)))
            else:
                incorrect_num = correct_num + max(1, abs(int(correct_num * 0.1)))

            # Format answers consistently
            if correct_num == int(correct_num):
                correct_answer = str(int(correct_num))
            else:
                correct_answer = str(correct_num)

            if incorrect_num == int(incorrect_num):
                incorrect_answer = str(int(incorrect_num))
            else:
                incorrect_answer = str(incorrect_num)

        except (ValueError, TypeError, IndexError) as e:
            print(f"Skipping item {i} due to answer parsing error: {answer} - {e}")
            continue

        asdiv_groups.append({
            'question': question,
            'correct_answer': correct_answer,
            'incorrect_answer': incorrect_answer,
            'dataset_type': 'asdiv'
        })

    if n_samples:
        asdiv_groups = asdiv_groups[:n_samples]

    print(f"Successfully loaded {len(asdiv_groups)} ASDiv questions")
    return asdiv_groups


def load_gsm8k_dataset(n_samples: int = None):
    """Load GSM8K dataset from HuggingFace"""
    from datasets import load_dataset
    import re
    import random

    print("Loading GSM8K dataset...")

    try:
        dataset = load_dataset("openai/gsm8k", "main")
        print("Successfully loaded GSM8K dataset")
    except Exception as e:
        print(f"Failed to load GSM8K dataset: {e}")
        raise

    # Try different split names
    split_names = ['test', 'train', 'validation']
    data = None

    for split_name in split_names:
        if split_name in dataset:
            data = dataset[split_name]
            print(f"Using {split_name} split")
            break

    if data is None:
        raise ValueError(f"No valid split found in dataset. Available splits: {list(dataset.keys())}")

    gsm8k_groups = []

    for i, item in enumerate(data):
        if i == 0:  # Print first item to debug structure
            print("Sample item structure:", {k: type(v) for k, v in item.items()})

        # GSM8K has 'question' and 'answer' fields
        question = item.get('question', '')
        answer = item.get('answer', '')

        if not question or not answer:
            continue

        # Extract numeric answer from the answer field
        try:
            # GSM8K answers typically end with "#### [number]"
            if "####" in answer:
                answer_text = answer.split("####")[-1].strip()
            else:
                answer_text = answer

            # Extract number from answer text
            numbers = re.findall(r'-?\d+\.?\d*', answer_text)
            if numbers:
                correct_num = float(numbers[0])
            else:
                print(f"Skipping item {i} - no number found in answer: {answer_text}")
                continue

            # Create a plausible incorrect answer (add/subtract a reasonable amount)
            if correct_num == 0:
                incorrect_num = 1
            elif correct_num > 0:
                if correct_num <= 10:
                    # For small numbers, add/subtract 1-3
                    incorrect_num = max(0, correct_num - random.randint(1, 3))
                else:
                    # For larger numbers, use percentage-based change
                    incorrect_num = max(0, correct_num - max(1, int(correct_num * 0.1)))
            else:
                incorrect_num = correct_num + max(1, abs(int(correct_num * 0.1)))

            # Format answers consistently
            if correct_num == int(correct_num):
                correct_answer = str(int(correct_num))
            else:
                correct_answer = str(correct_num)

            if incorrect_num == int(incorrect_num):
                incorrect_answer = str(int(incorrect_num))
            else:
                incorrect_answer = str(incorrect_num)

        except (ValueError, TypeError, IndexError) as e:
            print(f"Skipping item {i} due to answer parsing error: {answer} - {e}")
            continue

        gsm8k_groups.append({
            'question': question,
            'correct_answer': correct_answer,
            'incorrect_answer': incorrect_answer,
            'dataset_type': 'gsm8k'
        })

    if n_samples:
        gsm8k_groups = gsm8k_groups[:n_samples]

    print(f"Successfully loaded {len(gsm8k_groups)} GSM8K questions")
    return gsm8k_groups


def load_mmlu_dataset(n_samples: int = None):
    """Load MMLU dataset from HuggingFace"""
    from datasets import load_dataset
    import random

    print("Loading MMLU dataset...")

    try:
        dataset = load_dataset("cais/mmlu", "all")
        print("Successfully loaded MMLU dataset")
    except Exception as e:
        print(f"Failed to load MMLU dataset: {e}")
        raise

    # Try different split names
    split_names = ['test', 'validation', 'dev', 'train']
    data = None

    for split_name in split_names:
        if split_name in dataset:
            data = dataset[split_name]
            print(f"Using {split_name} split")
            break

    if data is None:
        raise ValueError(f"No valid split found in dataset. Available splits: {list(dataset.keys())}")

    mmlu_groups = []

    for i, item in enumerate(data):
        if i == 0:  # Print first item to debug structure
            print("Sample item structure:", {k: type(v) for k, v in item.items()})

        # MMLU has 'question', 'choices', and 'answer' fields
        question = item.get('question', '')
        choices = item.get('choices', [])
        answer = item.get('answer', None)

        if not question or not choices or answer is None:
            continue

        # Ensure we have 4 choices (A, B, C, D)
        if len(choices) != 4:
            continue

        try:
            # Answer is typically an integer (0, 1, 2, 3) corresponding to A, B, C, D
            if isinstance(answer, int) and 0 <= answer <= 3:
                correct_letter = ['A', 'B', 'C', 'D'][answer]
                correct_choice = choices[answer]

                # Create incorrect answers by selecting other choices
                incorrect_indices = [i for i in range(4) if i != answer]
                incorrect_index = random.choice(incorrect_indices)
                incorrect_letter = ['A', 'B', 'C', 'D'][incorrect_index]
                incorrect_choice = choices[incorrect_index]
            else:
                print(f"Skipping item {i} - invalid answer format: {answer}")
                continue

            # Format the question with choices
            formatted_question = f"{question}\n"
            for j, choice in enumerate(choices):
                formatted_question += f"{['A', 'B', 'C', 'D'][j]}. {choice}\n"
            formatted_question = formatted_question.strip()

        except (ValueError, TypeError, IndexError) as e:
            print(f"Skipping item {i} due to answer parsing error: {answer} - {e}")
            continue

        mmlu_groups.append({
            'question': formatted_question,
            'correct_answer': correct_letter,
            'incorrect_answer': incorrect_letter,
            'dataset_type': 'mmlu',
            'subject': item.get('subject', 'unknown')
        })

    if n_samples:
        mmlu_groups = mmlu_groups[:n_samples]

    print(f"Successfully loaded {len(mmlu_groups)} MMLU questions")
    return mmlu_groups


def load_dataset_by_choice(dataset_name: str, n_samples: int = None):
    """Load dataset based on user choice"""
    dataset_name = dataset_name.lower()

    if dataset_name == 'strategyqa':
        return load_strategyqa_dataset(n_samples)
    elif dataset_name == 'svamp':
        return load_svamp_dataset(n_samples)
    elif dataset_name == 'asdiv':
        return load_asdiv_dataset(n_samples)
    elif dataset_name == 'gsm8k':
        return load_gsm8k_dataset(n_samples)
    elif dataset_name == 'mmlu':
        return load_mmlu_dataset(n_samples)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'strategyqa', 'svamp', 'asdiv', 'gsm8k', or 'mmlu'")
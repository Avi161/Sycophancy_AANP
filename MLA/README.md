# Multi-Layer Activation Steering (MLAS) for Sycophancy Mitigation

This repository contains the implementation for our paper "Mitigating sycophancy in large language models via Sparse Activation Fusion and Multi-Layer Activation Steering". The code here specifically supports the Multi-Layer Activation Steering (MLAS) component of the work.

## Overview

This code implements Multi-Layer Activation Steering (MLAS), a method that identifies and removes "pressure directions" across multiple transformer layers to prevent models from capitulating under social pressure while preserving baseline accuracy.

## File Organization

```
├── main.py                    # Main execution script
├── config.py                  # Configuration and templates
├── model_utils.py            # Model setup utilities
├── data_utils.py             # Data loading for multiple datasets
├── tokenization.py           # Tokenization utilities
├── generation.py             # Text generation without intervention
├── intervention.py           # Core intervention logic and hooks
├── direction_finding.py      # Direction identification algorithms
├── evaluation.py             # Evaluation utilities
├── interactive.py            # Interactive evaluation interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your HuggingFace token in `model_utils.py` (line 12):
```python
login(token="your_huggingface_token_here")
```

## Quick Start

### Basic Usage

Run the complete analysis pipeline:
```bash
python main.py
```

This will:
1. Clone the sycophancy evaluation repository
2. Load and setup the Gemma-2-2B model
3. Find anti-sycophancy directions from training data
4. Evaluate the intervention on test data
5. Launch the interactive evaluation interface

### Interactive Evaluation

After running the main script, you can evaluate on multiple datasets:

```python
# Interactive menu
interactive_dataset_evaluation()

# Quick programmatic evaluation
quick_dataset_eval('strategyqa', 100, 0.15)  # dataset, n_questions, alpha
quick_dataset_eval('mmlu', 200, 0.15)
```

### Supported Datasets

- **StrategyQA**: Yes/No reasoning questions
- **SVAMP**: Math word problems  
- **ASDiv**: Academic math problems
- **GSM8K**: Grade school math problems
- **MMLU**: Multiple choice questions (A/B/C/D)

## Key Parameters

- `batch_size=1`: Critical for consistent direction finding (prevents cross-sequence interference)
- `alpha=0.2`: Conservative intervention strength for main evaluation
- `min_effect_size=0.4`: Threshold for selecting high-quality layer directions
- `n_samples=100`: Number of training samples for direction finding

## Core Components

### Direction Finding (`direction_finding.py`)

Identifies anti-sycophancy directions by contrasting:
- **Honest state**: Model's natural responses to neutral prompts
- **Sycophantic state**: Model's state after receiving strong user agreement

```python
layer_directions, stats = find_sycophancy_directions_fixed(
    model, train_groups, n_samples=100, batch_size=1
)
```

### Intervention (`intervention.py`)

Applies the intervention during generation:
```python
# Core intervention formula
intervention = alpha * proj_vectors
# Clip to prevent extreme values  
max_intervention = activation.abs().mean() * 0.5
intervention = torch.clamp(intervention, -max_intervention, max_intervention)
return activation + intervention
```

### Evaluation (`evaluation.py`)

Comprehensive evaluation across multiple metrics:
- Accuracy comparison (baseline vs intervention)
- Sycophancy reduction measures
- Cross-dataset generalization testing

## Results Reproduction

To reproduce our paper results:

1. **Sycophancy Evaluation**:
```bash
python main.py  # Runs evaluation on SycophancyEval dataset
```

2. **Cross-Dataset Evaluation**:
```python
# After main.py completes
interactive_dataset_evaluation()  # Choose option 6 for batch evaluation
```

Expected improvements:
- Reduces false admissions from 78% to 0%
- Maintains baseline accuracy (68% vs 70% unpressured)
- Shows modest performance trade-offs on unbiased datasets (3.75% average decrease)

## Key Implementation Details

### Batch Size = 1 Fix
The code uses `batch_size=1` during direction finding to prevent inconsistent results caused by:
- Padding effects when sequences have different lengths
- Cross-sequence attention interference
- Position-dependent activation extraction

### Intervention Clipping
```python
max_intervention = activation.abs().mean() * 0.5
intervention = torch.clamp(intervention, -max_intervention, max_intervention)
```
This prevents extreme activation modifications that could break generation.

### Multi-Layer Targeting
Focuses on layers 10+ where semantic processing occurs, recognizing that sycophancy is distributed across multiple layers rather than localized.

## Output Files

- `{dataset_name}_results.jsonl`: Detailed results for each evaluation
- Console output with layer-wise statistics and effect sizes
- Summary statistics for accuracy comparisons

## Troubleshooting

**Common Issues:**

1. **CUDA out of memory**: Reduce `n_samples` or use CPU
2. **Inconsistent results**: Ensure `batch_size=1` for direction finding
3. **Generation broken**: Lower `alpha` value (try 0.1)
4. **Dataset loading errors**: Check internet connection for HuggingFace datasets

**Effect Size Interpretation:**
- < 0.2: Very weak direction quality
- 0.2-0.5: Weak but usable
- 0.5-0.8: Moderate quality
- \> 0.8: Strong direction quality

## Citation

If you use this code, please cite our paper:
```
[Citation will be added upon acceptance]
```

# LLM Pruner Usage Guide

Complete guide for using the LLM Pruner toolkit.

## Installation

### From Source

```bash
git clone https://github.com/yourusername/llm_pruner.git
cd llm_pruner
pip install -e .
```

### From PyPI (when published)

```bash
pip install llm-pruner
```

## Three Ways to Use

### 1.  Command Line Interface (Easiest)

Prune a model with a single command:

```bash
llm-pruner \
  --model meta-llama/Llama-3.2-1B \
  --strategy smart \
  --eval-dataset eren23/pruner_eval \
  --output ./my_pruned_model \
  --visualize
```

**All CLI Options:**

```bash
llm-pruner --help

Options:
  --model MODEL                 HuggingFace model ID (required)
  --strategy {ablation,avss,pruneme,smart}
                               Pruning strategy (default: smart)
  --eval-dataset DATASET       HF dataset for evaluation (optional)
  --n-samples N                Number of eval samples (default: 80)
  --ratio RATIO                Prune ratio for avss/pruneme (default: 0.25)
  --threshold THRESHOLD        Threshold for ablation (default: -0.02)
  --tolerance TOLERANCE        Tolerance for smart pruning (default: 0.015)
  --output DIR                 Output directory (default: ./pruned_model)
  --export-yaml PATH           Export MergeKit config
  --visualize                  Generate plots
  --device {cuda,cpu}          Device to use
```

**Examples:**

```bash
# Quick pruning with AVSS
llm-pruner --model gpt2 --strategy avss --ratio 0.3

# Ablation with custom threshold
llm-pruner --model meta-llama/Llama-3.2-1B --strategy ablation --threshold -0.01

# Smart pruning with visualization
llm-pruner --model gpt2 --strategy smart --visualize --export-yaml config.yaml

# Use custom evaluation dataset
llm-pruner --model gpt2 --strategy smart --eval-dataset username/my-dataset
```

### 2.  Python API (Most Flexible)

#### Minimal Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_pruner import SmartPruner, load_dataset_from_hub

# Load
MODEL_ID = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).eval()

# Prune
eval_data = load_dataset_from_hub("eren23/pruner_eval")
pruner = SmartPruner(model, tokenizer)
scores = pruner.compute_layer_scores(eval_data)
candidates = pruner.select_layers(scores)
optimized = pruner.optimize_selection(candidates, eval_data)
pruned_model, _ = pruner.prune_and_save(optimized, "./pruned_model", MODEL_ID)
```

#### Full Control Example

```python
from llm_pruner import (
    AblationPruner, AVSSPruner, PruneMePruner, SmartPruner,
    ModelEvaluator, load_eval_dataset, plot_layer_importance
)

# Load your own evaluation data
eval_data = [
    {"prompt": "What is AI?", "reference": "Artificial Intelligence..."},
    {"prompt": "Explain Python", "reference": "Python is a programming language..."},
]

# Or load from multiple sources
eval_data = load_eval_dataset(n=120)

# Try different strategies
ablation = AblationPruner(model, tokenizer)
avss = AVSSPruner(model, tokenizer)
pruneme = PruneMePruner(model, tokenizer)

# Compare strategies
for pruner in [ablation, avss, pruneme]:
    scores = pruner.compute_layer_scores(eval_data)
    layers = pruner.select_layers(scores)
    print(f"{pruner.description()}: {layers}")

# Evaluate before/after
evaluator = ModelEvaluator(model, tokenizer)
before = evaluator.evaluate(eval_data)

pruned_model, _ = pruner.prune_and_save(layers, "./output", MODEL_ID)

evaluator_after = ModelEvaluator(pruned_model, tokenizer)
after = evaluator_after.evaluate(eval_data)

print(f"Score change: {after - before:+.4f}")
```

## Pruning Strategies in Detail

### 1. Ablation Pruning

**How it works:** Temporarily removes each layer and measures performance impact.

**When to use:**

- You want the most accurate pruning
- You have time for thorough analysis
- Quality > Speed

**Parameters:**

- `threshold`: Score drop allowed (default: -0.02 = 2% drop)

```python
pruner = AblationPruner(model, tokenizer)
scores = pruner.compute_layer_scores(eval_data)
layers = pruner.select_layers(scores, threshold=-0.01)  # Stricter
```

### 2. AVSS (Variance Analysis)

**How it works:** Measures variance in layer activations; low variance = redundant.

**When to use:**

- You want fast analysis
- You're pruning many layers
- Speed > Precision

**Parameters:**

- `ratio`: Fraction of layers to remove (default: 0.25)

```python
pruner = AVSSPruner(model, tokenizer)
scores = pruner.compute_layer_scores(eval_data)
layers = pruner.select_layers(scores, ratio=0.3)  # Remove 30%
```

### 3. PruneMe (Angular Distance)

**How it works:** Measures angle between input/output; small angle = redundant transformation.

**When to use:**

- Balanced speed/accuracy
- Geometric interpretation matters
- Middle ground approach

**Parameters:**

- `ratio`: Fraction to remove (default: 0.25)
- `threshold`: Optional distance threshold

```python
pruner = PruneMePruner(model, tokenizer)
scores = pruner.compute_layer_scores(eval_data)
layers = pruner.select_layers(scores, ratio=0.25)
```

### 4. Smart Pruner (Recommended)

**How it works:** Starts with ablation candidates, then iteratively tests combinations and "heals" problematic removals.

**When to use:**

- You want the best results
- You can afford extra compute
- Production-quality pruning

**Parameters:**

- `threshold`: Initial candidate selection (default: -0.02)
- `tolerance`: Max allowed drop during optimization (default: 0.015)

```python
pruner = SmartPruner(model, tokenizer)

# Phase 1: Find candidates
scores = pruner.compute_layer_scores(eval_data)
candidates = pruner.select_layers(scores, threshold=-0.02)

# Phase 2: Optimize
optimized = pruner.optimize_selection(
    candidates, 
    eval_data, 
    tolerance=0.01  # Only 1% drop allowed
)
```

## Advanced Usage

### Custom Evaluation Data

```python
# Load from HuggingFace dataset
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main", split="test")
eval_data = [
    {"prompt": x["question"], "reference": x["answer"]}
    for x in ds.select(range(100))
]
```

### Multi-Task Evaluation

```python
from llm_pruner import HolisticEvaluator

evaluator = HolisticEvaluator(model, tokenizer)
test_data = evaluator.load_diverse_data(n_per_task=100)
results = evaluator.run_benchmark(test_data)

# Results by task
for task, score in results.items():
    print(f"{task}: {score:.3f}")
```

### Visualizations

```python
from llm_pruner import (
    plot_layer_importance,
    plot_activation_comparison,
    plot_semantic_comparison
)

# 1. Importance heatmap
plot_layer_importance(scores, "Ablation Scores", "importance.png")

# 2. Activation flow
plot_activation_comparison(
    model, tokenizer, 
    "Test prompt here",
    pruned_layers=[5, 10, 15]
)

# 3. Semantic drift
plot_semantic_comparison(
    original_model, pruned_model, 
    tokenizer, "The capital of France is"
)
```

### Export for MergeKit

```python
# Generate config
pruner.export_mergekit_yaml(
    layers_to_remove=[3, 7, 11, 15],
    yaml_path="prune_config.yaml",
    source_model="meta-llama/Llama-3.2-1B"
)

# Then use with mergekit
# mergekit-yaml prune_config.yaml ./output --copy-tokenizer
```

### Manual Layer Selection

```python
# You choose exactly which layers
layers_to_remove = [5, 10, 15, 20, 25]

pruned_model, tokenizer = pruner.prune_and_save(
    layers_to_remove,
    output_dir="./custom_pruned",
    model_id=MODEL_ID
)
```

## Performance Tuning

### Memory Optimization

```python
# Use smaller batch size
from llm_pruner import ModelEvaluator

evaluator = ModelEvaluator(
    model, tokenizer, 
    batch_size=4,  # Reduce if OOM
    max_new_tokens=32  # Shorter generations
)
```

### Speed Optimization

```python
# Use fewer eval samples for fast iteration
eval_data = load_eval_dataset(n=20)  # Quick test

# Then do final run with more samples
eval_data_full = load_eval_dataset(n=120)
```

### Quality Optimization

```python
# Smart pruner with strict tolerance
pruner = SmartPruner(model, tokenizer)
scores = pruner.compute_layer_scores(eval_data)
candidates = pruner.select_layers(scores, threshold=-0.01)  # Strict
optimized = pruner.optimize_selection(candidates, eval_data, tolerance=0.005)  # Very strict
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
evaluator = ModelEvaluator(model, tokenizer, batch_size=2)

# Use CPU for evaluation
evaluator = ModelEvaluator(model, tokenizer, device="cpu")

# Use fewer samples
eval_data = eval_data[:20]
```

### Poor Pruning Results

```python
# Try stricter thresholds
layers = pruner.select_layers(scores, threshold=-0.01)  # Stricter

# Use SmartPruner
smart = SmartPruner(model, tokenizer)
optimized = smart.optimize_selection(candidates, eval_data, tolerance=0.005)

# Use more evaluation data
eval_data = load_eval_dataset(n=200)
```

### Unsupported Model Architecture

```python
# Check if model has expected attributes
if hasattr(model, "model"):
    print("Llama-style model")
    layers = model.model.layers
elif hasattr(model, "transformer"):
    print("GPT-style model")
    layers = model.transformer.h
else:
    print("Unsupported - may need custom implementation")
```

## Best Practices

1. **Start Small**: Test with 20-50 samples first
2. **Validate Results**: Always evaluate before/after
3. **Use Smart Pruner**: For production use
4. **Visualize**: Check activation plots for sanity
5. **Save Configs**: Export YAML for reproducibility
6. **Test Generations**: Don't just trust metrics
7. **Protect Critical Layers**: First/last 2 layers are kept by default

## Real-World Workflow

```python
# 1. Quick exploration with AVSS
avss = AVSSPruner(model, tokenizer)
quick_scores = avss.compute_layer_scores(eval_data[:20])
initial_candidates = avss.select_layers(quick_scores, ratio=0.3)

# 2. Refine with ablation
ablation = AblationPruner(model, tokenizer)
detailed_scores = ablation.compute_layer_scores(eval_data)
refined_candidates = ablation.select_layers(detailed_scores)

# 3. Optimize with SmartPruner
smart = SmartPruner(model, tokenizer)
final_layers = smart.optimize_selection(refined_candidates, eval_data, tolerance=0.01)

# 4. Prune and validate
pruned_model, _ = smart.prune_and_save(final_layers, "./final_model", MODEL_ID)

# 5. Comprehensive evaluation
holistic = HolisticEvaluator(pruned_model, tokenizer)
results = holistic.run_benchmark(holistic.load_diverse_data(100))
```

---

For more examples, see `example.py` and `quickstart.py` in the repo!

# LLM Pruner

**Optimize LLM depth with surgical precision. Pruneren combines state-of-the-art analysis, novel healing algorithms, and rich visualization to make models smaller and faster without sacrificing intelligence.**

## Features

- **Multiple Pruning Strategies**:

  - **Ablation Pruning**: Sensitivity-based layer removal
  - **AVSS**: Variance analysis for redundancy detection
  - **PruneMe**: Angular distance-based pruning (inspired by [Arcee-AI&#39;s PruneMe](https://github.com/arcee-ai/PruneMe))
  - **Smart Pruner**: Iterative optimization with automatic healing, SmartPruner features a novel self-healing algorithm that iteratively validates every cut in real-time, ensuring maximum parameter reduction without sacrificing cognitive performance.
- **Rich Evaluation**:

  - Semantic similarity scoring
  - Multi-task holistic benchmarking
  - Baseline vs pruned comparisons
- **Beautiful Visualizations**:

  - Layer importance heatmaps
  - Activation flow analysis
  - Token-level semantic consistency
  - Benchmark comparison charts
- **Simple API**: Prune models in just a few lines of code!

## Quick Start

### Installation

```bash
# From source
git clone https://github.com/yourusername/llm_pruner.git
cd llm_pruner
pip install -e .

# Or just install requirements
pip install -r requirements.txt
```

### Three Ways to Use

#### 1. Command Line (Easiest!)

```bash
llm-pruner \
  --model meta-llama/Llama-3.2-1B \
  --strategy smart \
  --eval-dataset eren23/pruner_eval \
  --output ./pruned_model \
  --visualize
```

See full CLI docs in [USAGE.md](USAGE.md).

#### 2. Python API (Most Flexible)

**Prune a Model in 30 Seconds**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_pruner import SmartPruner, load_dataset_from_hub

# Load model
MODEL_ID = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cuda",
    torch_dtype=torch.float16
).eval()

# Load evaluation data
eval_data = load_dataset_from_hub("eren23/pruner_eval")

# Prune!
pruner = SmartPruner(model, tokenizer)
scores = pruner.compute_layer_scores(eval_data)
candidates = pruner.select_layers(scores, threshold=-0.02)
optimized = pruner.optimize_selection(candidates, eval_data, tolerance=0.015)

# Save pruned model
pruned_model, _ = pruner.prune_and_save(optimized, "./pruned_model", MODEL_ID)
```

See `quickstart.py` for a complete minimal example.

## Usage Examples

### Example 1: Ablation Pruning

```python
from llm_pruner import AblationPruner, plot_layer_importance

# Initialize
pruner = AblationPruner(model, tokenizer)

# Compute scores (measures impact of removing each layer)
scores = pruner.compute_layer_scores(eval_data)

# Select layers with minimal impact
layers_to_remove = pruner.select_layers(scores, threshold=-0.02)

# Visualize
plot_layer_importance(scores, save_path="importance.png")

# Export for MergeKit
pruner.export_mergekit_yaml(layers_to_remove, "config.yaml", MODEL_ID)
```

### Example 2: AVSS (Variance-Based)

```python
from llm_pruner import AVSSPruner

# Find layers with lowest variance (least informative)
pruner = AVSSPruner(model, tokenizer)
scores = pruner.compute_layer_scores(eval_data)
layers = pruner.select_layers(scores, ratio=0.25)  # Remove 25% of layers
```

### Example 3: PruneMe (Angular Distance)

```python
from llm_pruner import PruneMePruner

# Find layers with smallest input-output distance (most redundant)
pruner = PruneMePruner(model, tokenizer)
scores = pruner.compute_layer_scores(eval_data)
layers = pruner.select_layers(scores, ratio=0.25)
```

### Example 4: Smart Pruning with Optimization

```python
from llm_pruner import SmartPruner

# Start with ablation candidates, then optimize iteratively
smart_pruner = SmartPruner(model, tokenizer)

# Initial candidates
scores = smart_pruner.compute_layer_scores(eval_data)
candidates = smart_pruner.select_layers(scores, threshold=-0.02)

# Iteratively test and "heal" if performance drops too much
optimized = smart_pruner.optimize_selection(
    candidates, 
    eval_data, 
    tolerance=0.015  # Max 1.5% score drop
)
```

### Example 5: Comprehensive Evaluation

```python
from llm_pruner import ModelEvaluator, HolisticEvaluator

# Simple evaluation
evaluator = ModelEvaluator(model, tokenizer)
score = evaluator.evaluate(eval_data)

# Multi-task benchmark
holistic = HolisticEvaluator(model, tokenizer)
test_data = holistic.load_diverse_data(n_per_task=100)
results = holistic.run_benchmark(test_data)
# Results: {"GSM8k": 0.73, "HellaSwag": 0.68, ...}
```

### Example 6: Visualizations

```python
from llm_pruner import (
    plot_layer_importance,
    plot_activation_comparison,
    plot_semantic_comparison,
    plot_holistic_benchmark
)

# Layer importance heatmap
plot_layer_importance(scores, "Layer Importance")

# Compare activations before/after pruning
plot_activation_comparison(model, tokenizer, "Test prompt", pruned_layers)

# Token-level semantic consistency
plot_semantic_comparison(original_model, pruned_model, tokenizer, "Test prompt")

# Benchmark comparison
plot_holistic_benchmark(results_orig, results_pruned, n_samples=100)
```

## Complete Example

See `example.py` for a comprehensive demo covering:

- All pruning strategies
- Evaluation comparisons
- All visualization types
- Saving and loading pruned models

## Evaluation Data

The toolkit supports multiple evaluation datasets:

### Load from scratch:

```python
from llm_pruner import load_eval_dataset

eval_data = load_eval_dataset(n=120)
# Samples from: GSM8k, ARC-Easy, SciQ, TruthfulQA, OpenBookQA, TriviaQA
```

### Upload to HuggingFace:

```python
from llm_pruner import upload_dataset_to_hub

upload_dataset_to_hub(eval_data, "username/my-eval-dataset")
```

### Load from HuggingFace:

```python
from llm_pruner import load_dataset_from_hub

eval_data = load_dataset_from_hub("username/my-eval-dataset")
```

## Advanced Features

### Manual Pruning

```python
# Get exact control
pruned_model, tokenizer = pruner.prune_and_save(
    layers_to_remove=[5, 10, 15, 20],
    output_dir="./my_pruned_model",
    model_id=MODEL_ID
)
```

### Custom Evaluation

```python
from llm_pruner import ModelEvaluator

# Use your own data
my_eval_data = [
    {"prompt": "Question 1", "reference": "Answer 1"},
    {"prompt": "Question 2", "reference": "Answer 2"},
]

evaluator = ModelEvaluator(model, tokenizer)
score = evaluator.evaluate(my_eval_data)
```

### MergeKit Integration

All pruners can export MergeKit-compatible YAML configs:

```python
pruner.export_mergekit_yaml(
    layers_to_remove=[3, 7, 11],
    yaml_path="prune_config.yaml",
    source_model=MODEL_ID
)
```

Then use with MergeKit:

```bash
mergekit-yaml prune_config.yaml ./output --copy-tokenizer --cuda
```

## Performance Tips

1. **Start Small**: Test with a small eval set first (n=20-50)
2. **Use Smart Pruner**: It automatically finds the optimal balance
3. **Protect Layers**: First 2 and last 2 layers are always kept
4. **Tolerance Tuning**: Start with 0.01-0.02 tolerance for SmartPruner
5. **Batch Size**: Adjust based on your GPU memory

## Pruning Strategies Compared

| Strategy              | Speed      | Accuracy   | Use Case             |
| --------------------- | ---------- | ---------- | -------------------- |
| **Ablation**    | ⭐⭐       | ⭐⭐⭐⭐⭐ | Best quality, slower |
| **AVSS**        | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | Fast screening       |
| **PruneMe**     | Fast       | High       | Balanced             |
| **SmartPruner** | Slow       | Highest    | Best results         |

## Example Results

Here's a real example of pruning Llama-3.2-1B (80 evaluation samples):

```
Model: meta-llama/Llama-3.2-1B (1,235,814,400 parameters)

EXAMPLE 1: Ablation Pruning
- Baseline Score: 0.2036
- Suggested layers to remove: [6, 8]

EXAMPLE 2: AVSS Pruning
- AVSS suggests removing: []  (model too small/uniform)

EXAMPLE 3: PruneMe (Angular Distance)
- PruneMe suggests removing: [11, 12, 13]

EXAMPLE 4: Smart Pruning (Optimization)
- Candidates: [6, 8]
- Tolerance: 1.5% drop allowed
- Baseline Score: 0.2036
- Optimized removal list: [6]
- Final Score: 0.2499 (Baseline: 0.2036)

EXAMPLE 5: Apply Pruning
- Baseline params: 1,235,814,400
- Pruned params:   1,174,992,896
- Reduction:       4.9%
- Baseline score:  0.2036
- Pruned score:    0.2499
- Delta:           +0.0464  (IMPROVEMENT!)

Total processing time: ~7 minutes on T4 GPU
```





The Smart Pruner successfully removed layer 6, resulting in a **4.9% parameter reduction** with a **+4.6% performance improvement** on the evaluation set. This demonstrates that some layers can actually harm model performance when their contribution is negative. Also ofc in many other pruning tests the performance was decreased, this one above is extremely cherry picked, only to demonstrate the potential :D

## Contributing

Contributions welcome! Areas for improvement:

- Additional pruning strategies
- More evaluation benchmarks
- Architecture support (currently supports Llama-style and GPT-style models)
- Quantization integration

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{llm_pruner,
  title = {LLM Pruner: A Universal Toolkit for Layer Pruning},
  author = {Eren Akbulut},
  year = {2024},
  url = {https://github.com/yourusername/llm_pruner}
}
```

## Acknowledgments

This toolkit is built upon and inspired by several important works in the field:

### Papers

1. **"The Unreasonable Ineffectiveness of the Deeper Layers"** - Gromov et al. (2024)

   - [arXiv:2403.17887](https://arxiv.org/abs/2403.17887)
   - Demonstrates that LLMs can have substantial layers removed with minimal performance loss
   - Foundation for our layer redundancy analysis approaches
2. **"Shorter is Better: Depth Pruning for Large Language Models"** - Bo et al. (2024)

   - [arXiv:2411.02117](https://arxiv.org/abs/2411.02117)
   - Provides insights into depth-based pruning strategies

### Open Source Projects

- **[MergeKit](https://github.com/arcee-ai/mergekit)** by Arcee AI

  - Used for model merging and layer manipulation
  - Our tool exports MergeKit-compatible configurations
- **[PruneMe](https://github.com/arcee-ai/PruneMe)** by Arcee AI

  - Inspiration for angular distance-based pruning approach
  - Block similarity computation methods

### Techniques

- Ablation studies in neural networks
- AVSS (Activation Variance Scoring System)
- Angular distance measurement for layer similarity
- Iterative optimization with performance healing

## License

MIT License - see LICENSE file for details.

## Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions

---

Made for the open source community

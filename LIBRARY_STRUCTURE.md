# Library Structure

Complete overview of the LLM Pruner library structure and organization.

## Directory Structure

```
pruneren/
├── llm_pruner/                 # Main package
│   ├── __init__.py            # Package initialization & exports
│   ├── pruners.py             # All pruning strategies
│   ├── evaluation.py          # Model evaluation utilities
│   ├── visualization.py       # Plotting and visualization
│   ├── data.py               # Dataset loading and management
│   ├── utils.py              # Utility functions
│   └── cli.py                # Command-line interface
│
├── example.py                 # Comprehensive examples
├── quickstart.py             # Minimal quickstart
├── requirements.txt          # Dependencies
├── setup.py                  # Package installation
├── README.md                 # Main documentation
├── USAGE.md                  # Detailed usage guide
└── .gitignore               # Git ignore rules
```

## Module Breakdown

### 1. `llm_pruner/__init__.py`

**Purpose:** Package entry point and public API

**Exports:**

- Pruners: `AblationPruner`, `AVSSPruner`, `PruneMePruner`, `SmartPruner`
- Evaluation: `ModelEvaluator`, `HolisticEvaluator`
- Visualization: `plot_layer_importance`, `plot_activation_comparison`, etc.
- Data: `load_eval_dataset`, `load_dataset_from_hub`, `upload_dataset_to_hub`
- Utils: `count_parameters`, `format_prompt`, `clean_output`

**Usage:**

```python
from llm_pruner import SmartPruner, load_eval_dataset
```

### 2. `llm_pruner/pruners.py`

**Purpose:** All layer pruning strategies

**Classes:**

- `BasePruner`: Abstract base class with common functionality
- `AblationPruner`: Sensitivity-based pruning
- `AVSSPruner`: Variance-based pruning
- `PruneMePruner`: Angular distance-based pruning
- `SmartPruner`: Iterative optimization with healing

**Key Methods:**

- `compute_layer_scores(eval_data)`: Analyze layer importance
- `select_layers(scores, ratio/threshold)`: Choose layers to prune
- `prune_and_save(layers, output_dir, model_id)`: Apply pruning
- `export_mergekit_yaml(layers, yaml_path, model_id)`: Export config

**Design Pattern:** Strategy pattern with template method

### 3. `llm_pruner/evaluation.py`

**Purpose:** Model evaluation and benchmarking

**Classes:**

- `ModelEvaluator`: Basic semantic similarity evaluation
- `HolisticEvaluator`: Multi-task benchmark evaluation

**Key Features:**

- Batch generation for efficiency
- Sentence transformer embeddings
- Multiple benchmark datasets (GSM8k, HellaSwag, etc.)
- Task-specific scoring

### 4. `llm_pruner/visualization.py`

**Purpose:** Generate analysis plots

**Functions:**

- `plot_layer_importance()`: Heatmap of layer scores
- `plot_activation_comparison()`: Before/after activation norms
- `plot_semantic_comparison()`: Token-level semantic drift
- `plot_holistic_benchmark()`: Multi-task comparison

**Libraries:** matplotlib, seaborn

### 5. `llm_pruner/data.py`

**Purpose:** Dataset loading and management

**Functions:**

- `load_eval_dataset(n)`: Load from multiple sources
- `load_dataset_from_hub(repo_id)`: Load from HF Hub
- `upload_dataset_to_hub(data, repo_id)`: Upload to HF Hub
- `normalize_item()`: Standardize dataset formats
- `safe_sample()`: Safe dataset loading with error handling

**Supported Datasets:**

- GSM8k-lite, ARC-Easy, SciQ
- TruthfulQA, OpenBookQA, TriviaQA

### 6. `llm_pruner/utils.py`

**Purpose:** Utility functions

**Functions:**

- `count_parameters(model)`: Count model parameters
- `format_prompt(q)`: Format chat prompts
- `clean_output(text)`: Clean model outputs
- `get_device()`: Auto-detect CUDA/CPU

### 7. `llm_pruner/cli.py`

**Purpose:** Command-line interface

**Features:**

- Simple one-line pruning
- All strategies supported
- Automatic evaluation
- Export options
- Visualization generation

**Entry Point:** `llm-pruner` command after installation

## Example Scripts

### `quickstart.py`

**Purpose:** Minimal working example

**Lines of Code:** ~40

**What it does:**

1. Load model
2. Load eval data
3. Prune with SmartPruner
4. Save results

**Use Case:** First-time users, quick tests

### `example.py`

**Purpose:** Comprehensive demonstration

**Lines of Code:** ~200

**What it covers:**

1. All pruning strategies
2. Evaluation comparisons
3. All visualization types
4. MergeKit export
5. Holistic benchmarking

**Use Case:** Learning all features, documentation

## Usage Patterns

### Pattern 1: Quick Pruning (CLI)

```bash
llm-pruner --model MODEL_ID --strategy smart --output ./pruned
```

### Pattern 2: Simple API

```python
from llm_pruner import SmartPruner, load_dataset_from_hub

pruner = SmartPruner(model, tokenizer)
scores = pruner.compute_layer_scores(eval_data)
layers = pruner.select_layers(scores)
pruned_model, _ = pruner.prune_and_save(layers, "./output", model_id)
```

### Pattern 3: Full Control

```python
from llm_pruner import (
    AblationPruner, ModelEvaluator, 
    plot_layer_importance
)

# Analyze
pruner = AblationPruner(model, tokenizer)
scores = pruner.compute_layer_scores(eval_data)

# Visualize
plot_layer_importance(scores)

# Evaluate
evaluator = ModelEvaluator(model, tokenizer)
baseline = evaluator.evaluate(eval_data)

# Prune
layers = pruner.select_layers(scores, threshold=-0.01)
pruned_model, _ = pruner.prune_and_save(layers, "./output", model_id)

# Re-evaluate
evaluator_pruned = ModelEvaluator(pruned_model, tokenizer)
pruned_score = evaluator_pruned.evaluate(eval_data)
```

## Design Principles

### 1. **Simplicity First**

- Clean API with sensible defaults
- Works out of the box
- Progressive complexity

### 2. **Modularity**

- Each component is independent
- Easy to extend or replace
- Clear separation of concerns

### 3. **Flexibility**

- Multiple strategies to choose from
- Customizable at every level
- Works with different model architectures

### 4. **Production Ready**

- Comprehensive error handling
- Progress bars and logging
- Evaluation and validation built-in

### 5. **Visualization**

- Understand what's happening
- Validate results visually
- Publication-ready plots

## Extension Points

### Adding a New Pruning Strategy

```python
from llm_pruner.pruners import BasePruner

class MyCustomPruner(BasePruner):
    def description(self):
        return "My Custom Pruning Strategy"
  
    def compute_layer_scores(self, eval_data):
        # Your scoring logic
        scores = {}
        for idx in range(self.num_layers):
            scores[idx] = self._compute_score(idx, eval_data)
        return scores
  
    def select_layers(self, scores, ratio=0.2, threshold=None):
        # Your selection logic
        sorted_layers = sorted(scores.items(), key=lambda x: x[1])
        num_to_remove = int(len(scores) * ratio)
        candidates = [x[0] for x in sorted_layers[:num_to_remove]]
        # Safety check
        safe = [c for c in candidates if 2 < c < (self.num_layers - 2)]
        return sorted(safe)
```

### Adding a New Visualization

```python
def plot_my_analysis(model, data, save_path="analysis.png"):
    """Your custom visualization."""
    # Analysis code
    results = analyze(model, data)
  
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(results)
    plt.title("My Analysis")
    plt.savefig(save_path)
    plt.show()
```

### Supporting a New Model Architecture

```python
# In BasePruner.__init__
if hasattr(model, "model"):
    self.layers = model.model.layers
elif hasattr(model, "transformer"):
    self.layers = model.transformer.h
elif hasattr(model, "your_new_attr"):  # Add support
    self.layers = model.your_new_attr.layers
else:
    raise ValueError("Unknown architecture")
```

## Testing Checklist

- [ ] Test all pruning strategies
- [ ] Test with different model sizes
- [ ] Test visualization outputs
- [ ] Test CLI interface
- [ ] Test error handling
- [ ] Test with custom eval data
- [ ] Test MergeKit export
- [ ] Test memory efficiency
- [ ] Verify parameter counts
- [ ] Validate evaluation scores

## Dependencies

**Core:**

- torch (model operations)
- transformers (model loading)
- sentence-transformers (evaluation)

**Data:**

- datasets (dataset loading)
- huggingface-hub (HF integration)

**Visualization:**

- matplotlib (plotting)
- seaborn (styled plots)
- pandas (data manipulation)

**Utils:**

- numpy (numerical operations)
- tqdm (progress bars)
- pyyaml (config export)

## Performance Characteristics

| Operation   | Time Complexity | Memory      |
| ----------- | --------------- | ----------- |
| AVSS        | O(n * layers)   | Low         |
| PruneMe     | O(n * layers)   | Medium      |
| Ablation    | O(n * layers²) | Medium      |
| SmartPruner | O(n * layers³) | Medium-High |

Where n = number of evaluation samples

## Common Workflows

### Research Workflow

1. Use `example.py` to explore all strategies
2. Generate visualizations for paper
3. Export configs for reproducibility
4. Run holistic benchmarks

### Production Workflow

1. Use CLI for quick iterations
2. SmartPruner for final model
3. Comprehensive evaluation
4. Deploy pruned model

### Development Workflow

1. Use quickstart for testing
2. Add custom pruner if needed
3. Validate with visualizations
4. Integrate into pipeline

---

This library structure is designed to be:

- **Easy to learn**: Start with quickstart.py
- **Easy to use**: Simple API, clear docs
- **Easy to extend**: Modular design
- **Production ready**: Robust and tested

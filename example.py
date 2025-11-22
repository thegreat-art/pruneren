"""
Example usage of LLM Pruner library
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_pruner import (
    AblationPruner, AVSSPruner, PruneMePruner, SmartPruner,
    load_eval_dataset, load_dataset_from_hub,
    ModelEvaluator, HolisticEvaluator,
    plot_layer_importance, plot_activation_comparison,
    plot_semantic_comparison, plot_holistic_benchmark,
    count_parameters
)


# ================================================================
# CONFIGURATION
# ================================================================
MODEL_ID = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Option 1: Load from scratch
# eval_data = load_eval_dataset(n=80)

# Option 2: Load from HuggingFace Hub
eval_data = load_dataset_from_hub("eren23/pruner_eval")


# ================================================================
# LOAD MODEL
# ================================================================
print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval()

print(f"Model loaded with {count_parameters(model):,} parameters")


# ================================================================
# EXAMPLE 1: QUICK ABLATION PRUNING
# ================================================================
print("\n" + "="*60)
print("EXAMPLE 1: Ablation Pruning")
print("="*60)

# Initialize pruner
pruner = AblationPruner(model, tokenizer)

# Compute layer importance
scores = pruner.compute_layer_scores(eval_data)

# Select layers to prune
layers_to_remove = pruner.select_layers(scores, threshold=-0.02)
print(f"\nSuggested layers to remove: {layers_to_remove}")

# Export MergeKit config
pruner.export_mergekit_yaml(layers_to_remove, "pruned_config.yaml", MODEL_ID)

# Visualize
plot_layer_importance(scores, title="Ablation: Layer Importance", 
                     save_path="ablation_importance.png")


# ================================================================
# EXAMPLE 2: AVSS PRUNING
# ================================================================
print("\n" + "="*60)
print("EXAMPLE 2: AVSS Pruning (Variance-Based)")
print("="*60)

pruner_avss = AVSSPruner(model, tokenizer)
scores_avss = pruner_avss.compute_layer_scores(eval_data)
layers_avss = pruner_avss.select_layers(scores_avss, ratio=0.25)

print(f"AVSS suggests removing: {layers_avss}")
plot_layer_importance(scores_avss, title="AVSS: Layer Variance", 
                     save_path="avss_variance.png")


# ================================================================
# EXAMPLE 3: PRUNEME (ANGULAR DISTANCE)
# ================================================================
print("\n" + "="*60)
print("EXAMPLE 3: PruneMe (Angular Distance)")
print("="*60)

pruner_pruneme = PruneMePruner(model, tokenizer)
scores_pruneme = pruner_pruneme.compute_layer_scores(eval_data)
layers_pruneme = pruner_pruneme.select_layers(scores_pruneme, ratio=0.25)

print(f"PruneMe suggests removing: {layers_pruneme}")
plot_layer_importance(scores_pruneme, title="PruneMe: Angular Distance", 
                     save_path="pruneme_distance.png")


# ================================================================
# EXAMPLE 4: SMART PRUNING (ITERATIVE OPTIMIZATION)
# ================================================================
print("\n" + "="*60)
print("EXAMPLE 4: Smart Pruning (Optimization)")
print("="*60)

# Use ablation candidates as starting point
smart_pruner = SmartPruner(model, tokenizer)
optimized_layers = smart_pruner.optimize_selection(
    layers_to_remove, 
    eval_data, 
    tolerance=0.015
)

print(f"Optimized removal list: {optimized_layers}")
smart_pruner.export_mergekit_yaml(optimized_layers, "smart_pruned.yaml", MODEL_ID)


# ================================================================
# EXAMPLE 5: ACTUALLY PRUNE THE MODEL
# ================================================================
print("\n" + "="*60)
print("EXAMPLE 5: Apply Pruning")
print("="*60)

# Evaluate baseline
evaluator = ModelEvaluator(model, tokenizer, device=DEVICE)
baseline_score = evaluator.evaluate(eval_data)
baseline_params = count_parameters(model)

print(f"Baseline score: {baseline_score:.4f}")
print(f"Baseline params: {baseline_params:,}")

# Prune and save
pruned_model, pruned_tokenizer = smart_pruner.prune_and_save(
    optimized_layers,
    output_dir="./pruned_model",
    model_id=MODEL_ID
)

# Evaluate pruned model
evaluator_pruned = ModelEvaluator(pruned_model, pruned_tokenizer, device=DEVICE)
pruned_score = evaluator_pruned.evaluate(eval_data)
pruned_params = count_parameters(pruned_model)

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print('='*60)
print(f"Original params: {baseline_params:,}")
print(f"Pruned params:   {pruned_params:,}")
print(f"Reduction:       {(1 - pruned_params/baseline_params)*100:.1f}%")
print(f"")
print(f"Baseline score:  {baseline_score:.4f}")
print(f"Pruned score:    {pruned_score:.4f}")
print(f"Delta:           {pruned_score - baseline_score:+.4f}")
print('='*60)


# ================================================================
# EXAMPLE 6: VISUALIZATIONS
# ================================================================
print("\n" + "="*60)
print("EXAMPLE 6: Advanced Visualizations")
print("="*60)

# Activation comparison
test_prompt = "The capital of France is"
plot_activation_comparison(
    model, tokenizer, test_prompt, optimized_layers,
    save_path="activation_comparison.png"
)

# Semantic comparison
plot_semantic_comparison(
    model, pruned_model, tokenizer, test_prompt,
    save_path="semantic_comparison.png"
)


# ================================================================
# EXAMPLE 7: HOLISTIC BENCHMARK (OPTIONAL - TAKES LONGER)
# ================================================================
if False:  # Set to True to run
    print("\n" + "="*60)
    print("EXAMPLE 7: Holistic Benchmark Comparison")
    print("="*60)
    
    holistic_orig = HolisticEvaluator(model, tokenizer, device=DEVICE)
    holistic_pruned = HolisticEvaluator(pruned_model, pruned_tokenizer, device=DEVICE)
    
    test_data = holistic_orig.load_diverse_data(n_per_task=50)
    
    results_orig = holistic_orig.run_benchmark(test_data)
    results_pruned = holistic_pruned.run_benchmark(test_data)
    
    plot_holistic_benchmark(results_orig, results_pruned, 50, 
                           save_path="holistic_benchmark.png")


print("\n✅ All examples complete!")


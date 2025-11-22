"""
Quickstart: Prune a model in just a few lines
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_pruner import SmartPruner, load_dataset_from_hub, count_parameters

# Configuration
MODEL_ID = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print(f"Loading {MODEL_ID}...")
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

# Load evaluation data
eval_data = load_dataset_from_hub("eren23/pruner_eval")

# Initialize pruner and compute scores
pruner = SmartPruner(model, tokenizer)
print("\nComputing layer importance scores...")
scores = pruner.compute_layer_scores(eval_data)

# Select layers to prune
candidates = pruner.select_layers(scores, threshold=-0.02)
print(f"\nInitial candidates: {candidates}")

# Optimize selection
optimized = pruner.optimize_selection(candidates, eval_data, tolerance=0.015)
print(f"Optimized selection: {optimized}")

# Prune and save
print("\nPruning model...")
pruned_model, _ = pruner.prune_and_save(optimized, "./pruned_model", MODEL_ID)

# Compare
orig_params = count_parameters(model)
pruned_params = count_parameters(pruned_model)

print(f"\n{'='*60}")
print(f"Original: {orig_params:,} parameters")
print(f"Pruned:   {pruned_params:,} parameters")
print(f"Reduced:  {(1 - pruned_params/orig_params)*100:.1f}%")
print(f"{'='*60}")
print("\n✅ Done! Pruned model saved to ./pruned_model")


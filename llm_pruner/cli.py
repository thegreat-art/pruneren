"""
Command-line interface for LLM Pruner
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .pruners import AblationPruner, AVSSPruner, PruneMePruner, SmartPruner
from .data import load_eval_dataset, load_dataset_from_hub
from .evaluation import ModelEvaluator
from .utils import count_parameters
from .visualization import plot_layer_importance


def main():
    parser = argparse.ArgumentParser(description="LLM Pruner - Prune language models easily")
    
    parser.add_argument("--model", type=str, required=True, 
                       help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B)")
    parser.add_argument("--strategy", type=str, default="smart",
                       choices=["ablation", "avss", "pruneme", "smart"],
                       help="Pruning strategy to use")
    parser.add_argument("--eval-dataset", type=str, default=None,
                       help="HuggingFace dataset ID for evaluation (optional)")
    parser.add_argument("--n-samples", type=int, default=80,
                       help="Number of evaluation samples")
    parser.add_argument("--ratio", type=float, default=0.25,
                       help="Fraction of layers to prune (for avss/pruneme)")
    parser.add_argument("--threshold", type=float, default=-0.02,
                       help="Threshold for ablation pruning")
    parser.add_argument("--tolerance", type=float, default=0.015,
                       help="Tolerance for smart pruning")
    parser.add_argument("--output", type=str, default="./pruned_model",
                       help="Output directory for pruned model")
    parser.add_argument("--export-yaml", type=str, default=None,
                       help="Export MergeKit YAML config to this path")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Setup device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()
    
    orig_params = count_parameters(model)
    print(f"Model loaded: {orig_params:,} parameters")
    
    # Load evaluation data
    print(f"\nLoading evaluation data...")
    if args.eval_dataset:
        eval_data = load_dataset_from_hub(args.eval_dataset)
    else:
        eval_data = load_eval_dataset(n=args.n_samples)
    
    # Initialize pruner
    print(f"\nInitializing {args.strategy} pruner...")
    if args.strategy == "ablation":
        pruner = AblationPruner(model, tokenizer)
    elif args.strategy == "avss":
        pruner = AVSSPruner(model, tokenizer)
    elif args.strategy == "pruneme":
        pruner = PruneMePruner(model, tokenizer)
    else:  # smart
        pruner = SmartPruner(model, tokenizer)
    
    # Compute scores
    print(f"\nComputing layer importance scores...")
    scores = pruner.compute_layer_scores(eval_data)
    
    # Select layers
    if args.strategy in ["avss", "pruneme"]:
        layers_to_remove = pruner.select_layers(scores, ratio=args.ratio)
    elif args.strategy == "ablation":
        layers_to_remove = pruner.select_layers(scores, threshold=args.threshold)
    else:  # smart
        candidates = pruner.select_layers(scores, threshold=args.threshold)
        print(f"\nRunning optimization (tolerance={args.tolerance})...")
        layers_to_remove = pruner.optimize_selection(candidates, eval_data, 
                                                     tolerance=args.tolerance)
    
    print(f"\nSelected {len(layers_to_remove)} layers to remove: {layers_to_remove}")
    
    # Export YAML if requested
    if args.export_yaml:
        pruner.export_mergekit_yaml(layers_to_remove, args.export_yaml, args.model)
    
    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        plot_layer_importance(scores, title=f"{args.strategy.upper()} Layer Importance",
                            save_path=f"{args.strategy}_importance.png")
    
    # Evaluate baseline
    print("\nEvaluating baseline model...")
    evaluator = ModelEvaluator(model, tokenizer, device=device)
    baseline_score = evaluator.evaluate(eval_data)
    print(f"Baseline score: {baseline_score:.4f}")
    
    # Prune and save
    print(f"\nPruning model and saving to {args.output}...")
    pruned_model, _ = pruner.prune_and_save(layers_to_remove, args.output, args.model)
    
    # Evaluate pruned model
    print("\nEvaluating pruned model...")
    evaluator_pruned = ModelEvaluator(pruned_model, tokenizer, device=device)
    pruned_score = evaluator_pruned.evaluate(eval_data)
    
    pruned_params = count_parameters(pruned_model)
    
    # Print summary
    print("\n" + "="*70)
    print("PRUNING SUMMARY")
    print("="*70)
    print(f"Strategy:        {args.strategy}")
    print(f"Layers removed:  {len(layers_to_remove)}")
    print(f"")
    print(f"Original params: {orig_params:,}")
    print(f"Pruned params:   {pruned_params:,}")
    print(f"Reduction:       {(1 - pruned_params/orig_params)*100:.1f}%")
    print(f"")
    print(f"Baseline score:  {baseline_score:.4f}")
    print(f"Pruned score:    {pruned_score:.4f}")
    print(f"Delta:           {pruned_score - baseline_score:+.4f}")
    print("="*70)
    print(f"\nDone! Pruned model saved to: {args.output}")
    

if __name__ == "__main__":
    main()


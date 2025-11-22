"""
Visualization utilities for pruning analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from typing import Dict, List
from tqdm import tqdm


def plot_layer_importance(scores: Dict[int, float], title: str = "Layer Importance", 
                          save_path: str = "layer_importance.png"):
    """
    Plot layer importance scores as a heatmap.
    
    Args:
        scores: Dictionary mapping layer index to importance score
        title: Plot title
        save_path: Where to save the plot
    """
    score_list = [scores.get(i, 0) for i in range(len(scores))]
    
    plt.figure(figsize=(14, 2))
    sns.heatmap(
        np.array([score_list]),
        cmap="coolwarm",
        annot=False,
        cbar=True,
        xticklabels=list(scores.keys()),
        yticklabels=["importance"]
    )
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved: {save_path}")


def plot_activation_comparison(model, tokenizer, prompt: str, pruned_layers: List[int],
                               save_path: str = "activation_comparison.png"):
    """
    Compare activation norms before and after pruning.
    
    Args:
        model: The model to analyze
        tokenizer: Model tokenizer
        prompt: Test prompt
        pruned_layers: List of layer indices that were pruned
        save_path: Where to save the plot
    """
    # Get layers
    if hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.transformer.h
    
    def scan_activations(pruned_indices):
        norms = []
        
        def record_hook(idx):
            def hook(module, args, output):
                if idx in pruned_indices:
                    hidden_state = args[0]
                    norms.append(hidden_state.norm(p=2, dim=-1).mean().item())
                    if isinstance(output, tuple):
                        return (args[0],) + output[1:]
                    return args[0]
                
                if isinstance(output, tuple):
                    hidden_state = output[0]
                else:
                    hidden_state = output
                
                norms.append(hidden_state.detach().norm(p=2, dim=-1).mean().item())
            return hook
        
        handles = [layers[i].register_forward_hook(record_hook(i)) for i in range(len(layers))]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        
        with torch.no_grad():
            model(**inputs)
        
        for h in handles:
            h.remove()
        
        return norms
    
    print("Scanning original model...")
    original_norms = scan_activations([])
    
    print("Scanning with pruned layers...")
    pruned_norms = scan_activations(pruned_layers)
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="darkgrid")
    
    plt.plot(original_norms, label="Original Model", color="blue", alpha=0.6, linewidth=2)
    plt.plot(pruned_norms, label="Pruned Model", color="red", linestyle="--", alpha=0.8, linewidth=2)
    
    for layer_idx in pruned_layers:
        plt.axvline(x=layer_idx, color='red', alpha=0.1)
    
    plt.title("Model Activation Norms by Layer", fontsize=14)
    plt.xlabel("Layer Index")
    plt.ylabel("Mean L2 Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved: {save_path}")


def plot_semantic_comparison(original_model, pruned_model, tokenizer, prompt: str,
                             save_path: str = "semantic_comparison.png"):
    """
    Compare semantic consistency between original and pruned models.
    
    Args:
        original_model: Original model
        pruned_model: Pruned model
        tokenizer: Model tokenizer
        prompt: Test prompt
        save_path: Where to save the plot
    """
    device = original_model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    with torch.no_grad():
        out_orig = original_model(**inputs, output_hidden_states=True)
        out_pruned = pruned_model(**inputs, output_hidden_states=True)
    
    # Token-by-token similarity
    h_orig = out_orig.hidden_states[-1].squeeze(0)
    h_pruned = out_pruned.hidden_states[-1].squeeze(0)
    
    h_orig = F.normalize(h_orig, p=2, dim=-1)
    h_pruned = F.normalize(h_pruned, p=2, dim=-1)
    
    similarities = (h_orig * h_pruned).sum(dim=-1).cpu().numpy()
    
    # Entropy (confidence)
    probs_orig = F.softmax(out_orig.logits[0], dim=-1)
    probs_pruned = F.softmax(out_pruned.logits[0], dim=-1)
    
    ent_orig = -(probs_orig * torch.log(probs_orig + 1e-9)).sum(dim=-1).cpu().numpy()
    ent_pruned = -(probs_pruned * torch.log(probs_pruned + 1e-9)).sum(dim=-1).cpu().numpy()
    
    tokens = [tokenizer.decode([t]) for t in tokenizer.encode(prompt)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Semantic consistency
    sns.lineplot(x=range(len(similarities)), y=similarities, ax=ax1, 
                marker="o", color="green", linewidth=2.5)
    ax1.set_title(f"Semantic Consistency per Token\nPrompt: '{prompt[:40]}...'", fontsize=14)
    ax1.set_ylabel("Similarity (1.0 = Identical)")
    ax1.set_ylim(0.8, 1.05)
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=45, ha="right")
    ax1.axhline(1.0, color='gray', linestyle='--')
    
    # Plot 2: Confidence
    sns.lineplot(x=range(len(ent_orig)), y=ent_orig, ax=ax2, 
                label="Original", color="blue", alpha=0.6)
    sns.lineplot(x=range(len(ent_pruned)), y=ent_pruned, ax=ax2, 
                label="Pruned", color="red", linestyle="--")
    
    ax2.set_title("Model Confidence (Entropy) - Lower is More Confident", fontsize=14)
    ax2.set_ylabel("Entropy")
    ax2.set_xticks(range(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=45, ha="right")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved: {save_path}")


def plot_holistic_benchmark(results_orig: Dict[str, float], results_pruned: Dict[str, float],
                           n_samples: int, save_path: str = "holistic_benchmark.png"):
    """
    Visualize results from holistic benchmark comparison.
    
    Args:
        results_orig: Benchmark results for original model
        results_pruned: Benchmark results for pruned model
        n_samples: Number of samples per task
        save_path: Where to save the plot
    """
    tasks = list(results_orig.keys())
    
    data = []
    for t in tasks:
        data.append({"Task": t, "Model": "Original", "Score": results_orig[t]})
        data.append({"Task": t, "Model": "Pruned", "Score": results_pruned[t]})
    
    df = pd.DataFrame(data)
    
    avg_orig = np.mean(list(results_orig.values()))
    avg_pruned = np.mean(list(results_pruned.values()))
    delta = avg_pruned - avg_orig
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(data=df, x="Task", y="Score", hue="Model", 
                    palette={"Original": "#4c72b0", "Pruned": "#c44e52"})
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.title(f"Pruning Impact Analysis (N={n_samples}/task)", fontsize=16, pad=20)
    plt.ylabel("Semantic Similarity Score", fontsize=12)
    plt.xlabel("Dataset / Task", fontsize=12)
    plt.ylim(0, max(df["Score"].max() * 1.15, 1.0))
    
    color = "green" if delta >= 0 else "red"
    symbol = "+" if delta >= 0 else ""
    summary_text = (
        f"Original Avg: {avg_orig:.4f}\n"
        f"Pruned Avg:   {avg_pruned:.4f}\n"
        f"Delta:       {symbol}{delta:.4f}"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.95, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', horizontalalignment='right', bbox=props, 
           color=color, fontweight='bold')
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved: {save_path}")


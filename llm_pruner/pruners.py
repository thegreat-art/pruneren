"""
Layer pruning strategies for LLMs
"""

import torch
import yaml
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from .evaluation import ModelEvaluator


class BasePruner(ABC):
    """Base class for all pruning strategies."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        if hasattr(model, "model"):
            self.layers = model.model.layers
        elif hasattr(model, "transformer"):
            self.layers = model.transformer.h
        else:
            raise ValueError("Unknown architecture - cannot identify layers")
        
        self.num_layers = len(self.layers)
    
    @abstractmethod
    def compute_layer_scores(self, eval_data: List[Dict[str, str]]) -> Dict[int, float]:
        """
        Compute importance scores for each layer.
        
        Returns:
            Dictionary mapping layer index to score
        """
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Return description of pruning strategy."""
        pass
    
    @abstractmethod
    def select_layers(self, scores: Dict[int, float], ratio: float = 0.2, 
                     threshold: Optional[float] = None) -> List[int]:
        """
        Select layers to prune based on scores.
        
        Args:
            scores: Layer importance scores
            ratio: Fraction of layers to prune
            threshold: Optional threshold for selection
            
        Returns:
            List of layer indices to prune
        """
        pass
    
    def export_mergekit_yaml(self, layers_to_remove: List[int], yaml_path: str, source_model: str):
        """Export pruning configuration as MergeKit YAML."""
        layers_to_remove = sorted(list(set(layers_to_remove)))
        slices = []
        start = 0
        
        for idx in layers_to_remove:
            if idx > start:
                slices.append({"start": start, "end": idx})
            start = idx + 1
        
        if start < self.num_layers:
            slices.append({"start": start, "end": self.num_layers})
        
        config = {
            "merge_method": "passthrough",
            "dtype": "float16",
            "slices": [
                {"sources": [{"model": source_model, "layer_range": [s["start"], s["end"]]}]}
                for s in slices
            ]
        }
        
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        print(f"MergeKit YAML saved: {yaml_path}")
    
    def prune_and_save(self, layers_to_remove: List[int], output_dir: str, 
                       model_id: str) -> tuple:
        """
        Prune layers from model and save.
        
        Args:
            layers_to_remove: List of layer indices to remove
            output_dir: Directory to save pruned model
            model_id: Original model ID
            
        Returns:
            Tuple of (pruned_model, tokenizer)
        """
        import os
        from transformers import AutoModelForCausalLM
        
        print(f"🔪 Pruning {len(layers_to_remove)} layers...")
        
        # Reload model for pruning
        device = self.model.device
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Identify layers to keep
        keep_layers = sorted(set(range(self.num_layers)) - set(layers_to_remove))
        print(f"Keeping {len(keep_layers)} layers")
        
        # Detect architecture
        if hasattr(base_model, "model"):
            layer_obj = base_model.model.layers
            container_type = "model.layers"
        elif hasattr(base_model, "transformer"):
            layer_obj = base_model.transformer.h
            container_type = "transformer.h"
        else:
            raise RuntimeError("Unknown model architecture")
        
        # Create new layer list
        new_layers = torch.nn.ModuleList([layer_obj[i] for i in keep_layers])
        
        # Replace layers
        if container_type == "model.layers":
            base_model.model.layers = new_layers
        else:
            base_model.transformer.h = new_layers
        
        # Fix layer_idx attributes for cache compatibility
        for new_idx, layer in enumerate(new_layers):
            # Update layer_idx in self_attn if it exists
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = new_idx
            # Also check for other attention mechanisms
            if hasattr(layer, "attn") and hasattr(layer.attn, "layer_idx"):
                layer.attn.layer_idx = new_idx
        
        # Fix config
        cfg = base_model.config
        new_count = len(keep_layers)
        
        for name in ["num_hidden_layers", "n_layers", "num_layers"]:
            if hasattr(cfg, name):
                setattr(cfg, name, new_count)
        
        print(f"Saving to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        base_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("Pruning complete!")
        return base_model, self.tokenizer


class AblationPruner(BasePruner):
    """Sensitivity-based pruning via layer ablation."""
    
    def description(self) -> str:
        return "Ablation (Sensitivity Analysis)"
    
    def skip_layer(self, module, args, out):
        """Hook to bypass a layer."""
        if isinstance(out, tuple):
            return (args[0],) + out[1:]
        return args[0]
    
    def compute_layer_scores(self, eval_data: List[Dict[str, str]]) -> Dict[int, float]:
        """Compute sensitivity scores by ablating each layer."""
        prompts = [x["prompt"] for x in eval_data]
        refs = [x["reference"] for x in eval_data]
        
        evaluator = ModelEvaluator(self.model, self.tokenizer, device=self.model.device)
        
        print("[Ablation] Establishing Baseline...")
        baseline = evaluator.evaluate(eval_data)
        print(f"Baseline Score: {baseline:.4f}")
        
        deltas = {}
        for idx in tqdm(range(self.num_layers), desc="Ablating layers"):
            hook = self.layers[idx].register_forward_hook(self.skip_layer)
            
            try:
                score = evaluator.evaluate(eval_data)
                deltas[idx] = score - baseline
            except Exception as e:
                print(f"Error ablating layer {idx}: {e}")
                deltas[idx] = -999.0
            
            hook.remove()
        
        return deltas
    
    def select_layers(self, scores: Dict[int, float], ratio: float = 0.2, 
                     threshold: Optional[float] = -0.02) -> List[int]:
        """Select layers with minimal impact (delta > threshold)."""
        candidates = []
        
        print("\n--- Ablation Results ---")
        print(f"{'Layer':<6} | {'Delta':<10} | {'Status'}")
        print("-" * 40)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for idx, delta in sorted_scores:
            status = "KEEP"
            if delta > threshold:
                status = "PRUNE"
                candidates.append(idx)
            
            if delta > 0 or delta < -0.05:
                print(f"{idx:<6} | {delta:+.4f}     | {status}")
        
        # Safety: protect first 2 and last 2 layers
        safe_candidates = [c for c in candidates if 2 < c < (self.num_layers - 2)]
        return sorted(safe_candidates)


class AVSSPruner(BasePruner):
    """Variance-based pruning (AVSS)."""
    
    def description(self) -> str:
        return "AVSS (Variance Analysis)"
    
    def compute_layer_scores(self, eval_data: List[Dict[str, str]]) -> Dict[int, float]:
        """Compute variance scores for each layer."""
        variances = {i: [] for i in range(self.num_layers)}
        
        def get_hook(layer_idx):
            def hook(module, args, output):
                x_out = output[0].detach() if isinstance(output, tuple) else output.detach()
                var = torch.var(x_out, dim=-1).mean().item()
                variances[layer_idx].append(var)
            return hook
        
        print(f"[AVSS] Analyzing {len(eval_data)} samples...")
        handles = [self.layers[i].register_forward_hook(get_hook(i)) 
                  for i in range(self.num_layers)]
        
        for x in tqdm(eval_data, desc="Computing variances"):
            inp = self.tokenizer(x["prompt"], return_tensors="pt").to(self.model.device)
            if "token_type_ids" in inp:
                del inp["token_type_ids"]
            with torch.no_grad():
                self.model(**inp)
        
        for h in handles:
            h.remove()
        
        return {k: np.mean(v) for k, v in variances.items()}
    
    def select_layers(self, scores: Dict[int, float], ratio: float = 0.25, 
                     threshold: Optional[float] = None) -> List[int]:
        """Select layers with lowest variance."""
        sorted_layers = sorted(scores.items(), key=lambda x: x[1])
        num_to_remove = int(len(scores) * ratio)
        candidates = [x[0] for x in sorted_layers[:num_to_remove]]
        
        # Safety
        safe_candidates = [c for c in candidates if 2 < c < (self.num_layers - 2)]
        return sorted(safe_candidates)


class PruneMePruner(BasePruner):
    """Angular distance-based pruning."""
    
    def description(self) -> str:
        return "PruneMe (Angular Distance)"
    
    def compute_layer_scores(self, eval_data: List[Dict[str, str]]) -> Dict[int, float]:
        """Compute angular distances for each layer."""
        distances = {i: [] for i in range(self.num_layers)}
        
        def get_hook(layer_idx):
            def hook(module, args, output):
                x_in = args[0].detach()
                x_out = output[0].detach() if isinstance(output, tuple) else output.detach()
                
                norm_in = F.normalize(x_in, p=2, dim=-1)
                norm_out = F.normalize(x_out, p=2, dim=-1)
                
                cos_sim = (norm_in * norm_out).sum(dim=-1)
                dist = 1.0 - cos_sim.mean()
                distances[layer_idx].append(dist.item())
            return hook
        
        print(f"[PruneMe] Analyzing {len(eval_data)} samples...")
        handles = [self.layers[i].register_forward_hook(get_hook(i)) 
                  for i in range(self.num_layers)]
        
        for x in tqdm(eval_data, desc="Computing distances"):
            inp = self.tokenizer(x["prompt"], return_tensors="pt").to(self.model.device)
            if "token_type_ids" in inp:
                del inp["token_type_ids"]
            with torch.no_grad():
                self.model(**inp)
        
        for h in handles:
            h.remove()
        
        return {k: np.mean(v) for k, v in distances.items()}
    
    def select_layers(self, scores: Dict[int, float], ratio: float = 0.25, 
                     threshold: Optional[float] = None) -> List[int]:
        """Select layers with smallest angular distance."""
        sorted_layers = sorted(scores.items(), key=lambda x: x[1])
        num_to_remove = int(len(scores) * ratio)
        
        if threshold is not None and 0 < threshold < 1:
            candidates = [k for k, v in scores.items() if v < threshold]
        else:
            candidates = [x[0] for x in sorted_layers[:num_to_remove]]
        
        # Safety
        safe_candidates = [c for c in candidates if 2 < c < (self.num_layers - 2)]
        return sorted(safe_candidates)


class SmartPruner(AblationPruner):
    """Iterative optimization-based pruning with healing."""
    
    def description(self) -> str:
        return "Smart Pruner (Iterative Optimization)"
    
    def optimize_selection(self, candidates: List[int], eval_data: List[Dict[str, str]], 
                          tolerance: float = 0.01) -> List[int]:
        """
        Iteratively test removal of candidate layers and heal if needed.
        
        Args:
            candidates: Initial candidate layers to prune
            eval_data: Evaluation dataset
            tolerance: Maximum allowed performance drop
            
        Returns:
            Optimized list of layers safe to remove
        """
        print(f"\nStarting SMART OPTIMIZATION")
        print(f"Candidates: {candidates}")
        print(f"Tolerance: Drop of {tolerance*100}% allowed")
        
        evaluator = ModelEvaluator(self.model, self.tokenizer, device=self.model.device)
        baseline = evaluator.evaluate(eval_data)
        print(f"Baseline Score: {baseline:.4f}")
        
        skipped_layers = set()
        
        def dynamic_skip_hook(layer_idx):
            def hook(module, args, output):
                if layer_idx in skipped_layers:
                    if isinstance(output, tuple):
                        return (args[0],) + output[1:]
                    return args[0]
                return output
            return hook
        
        # Attach hooks to all layers
        handles = [self.layers[i].register_forward_hook(dynamic_skip_hook(i)) 
                  for i in range(self.num_layers)]
        
        final_removal_list = []
        current_score = baseline
        
        pbar = tqdm(candidates, desc="Optimizing")
        for layer_idx in pbar:
            skipped_layers.add(layer_idx)
            
            try:
                new_score = evaluator.evaluate(eval_data)
                drop = baseline - new_score
                
                if drop < tolerance:
                    final_removal_list.append(layer_idx)
                    current_score = new_score
                    pbar.set_postfix({"Status": "PRUNED", "Score": f"{new_score:.3f}"})
                else:
                    skipped_layers.remove(layer_idx)
                    pbar.set_postfix({"Status": "RESTORED", "Score": f"{current_score:.3f}"})
            except Exception as e:
                print(f"Error testing layer {layer_idx}: {e}")
                skipped_layers.remove(layer_idx)
        
        for h in handles:
            h.remove()
        
        print("\nOPTIMIZATION COMPLETE")
        print(f"Original Candidates: {len(candidates)}")
        print(f"Safe to Remove:      {len(final_removal_list)}")
        print(f"Final Score:         {current_score:.4f} (Baseline: {baseline:.4f})")
        
        return sorted(final_removal_list)


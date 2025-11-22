"""
Model evaluation utilities
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

from .utils import format_prompt, clean_output


class ModelEvaluator:
    """Basic model evaluator using semantic similarity."""
    
    def __init__(self, model, tokenizer, device="cuda", batch_size=8, max_new_tokens=64):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    def evaluate(self, eval_data: List[Dict[str, str]]) -> float:
        """
        Evaluate model on dataset using semantic similarity.
        
        Args:
            eval_data: List of dicts with 'prompt' and 'reference' keys
            
        Returns:
            Mean cosine similarity score
        """
        prompts = [x["prompt"] for x in eval_data]
        references = [x["reference"] for x in eval_data]
        
        generated = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch = [format_prompt(x) for x in prompts[i:i+self.batch_size]]
            inp = self.tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True).to(self.device)
            
            if "token_type_ids" in inp:
                del inp["token_type_ids"]
            
            with torch.no_grad():
                out = self.model.generate(
                    **inp,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            decoded = self.tokenizer.batch_decode(out, skip_special_tokens=False)
            generated.extend([clean_output(x) for x in decoded])
        
        emb_gen = self.embedder.encode(generated, convert_to_tensor=True)
        emb_ref = self.embedder.encode(references, convert_to_tensor=True)
        
        return util.cos_sim(emb_gen, emb_ref).diagonal().mean().item()


class HolisticEvaluator:
    """Comprehensive evaluator across multiple benchmark tasks."""
    
    def __init__(self, model, tokenizer, device="cuda", batch_size=8, max_new_tokens=64):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    def load_diverse_data(self, n_per_task: int = 30) -> List[Dict[str, str]]:
        """Load diverse evaluation data from multiple benchmarks."""
        from datasets import load_dataset
        
        print(f"Loading {n_per_task} samples per task...")
        eval_set = []
        
        # GSM8k (Reasoning)
        try:
            ds = load_dataset("openai/gsm8k", "main", split="test")
            for x in ds.select(range(min(n_per_task, len(ds)))):
                eval_set.append({"task": "GSM8k", "prompt": x["question"], "ref": x["answer"]})
        except Exception as e:
            print(f"⚠️  GSM8k failed to load: {e}")
        
        # HellaSwag (Common Sense)
        try:
            ds = load_dataset("rowan/hellaswag", split="validation")
            for x in ds.select(range(min(n_per_task, len(ds)))):
                ctx = x["ctx"]
                correct_ending = x["endings"][int(x["label"])]
                eval_set.append({"task": "HellaSwag", "prompt": ctx, "ref": correct_ending})
        except Exception as e:
            print(f"HellaSwag failed to load: {e}")
        
        print(f"Loaded {len(eval_set)} total diverse samples")
        return eval_set
    
    def run_benchmark(self, eval_set: List[Dict[str, str]]) -> Dict[str, float]:
        """Run comprehensive benchmark across all tasks."""
        print("🚀 Running Holistic Benchmark...")
        
        # Group by task
        tasks = defaultdict(list)
        for item in eval_set:
            tasks[item["task"]].append(item)
        
        final_report = {}
        
        for task_name, items in tasks.items():
            prompts = [self._fmt(x["prompt"]) for x in items]
            refs = [x["ref"] for x in items]
            
            # Generate
            preds = self._generate_batch(prompts)
            
            # Score
            emb_pred = self.embedder.encode(preds, convert_to_tensor=True)
            emb_ref = self.embedder.encode(refs, convert_to_tensor=True)
            score = util.cos_sim(emb_pred, emb_ref).diagonal().mean().item()
            
            final_report[task_name] = score
            print(f"   👉 {task_name}: {score:.4f}")
        
        avg_score = np.mean(list(final_report.values()))
        print(f"\nCOMPOSITE SCORE: {avg_score:.4f}")
        return final_report
    
    def _fmt(self, p: str) -> str:
        """Format prompt."""
        return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>"
    
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, 
                               truncation=True).to(self.device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=False)
        return [clean_output(d) for d in decoded]


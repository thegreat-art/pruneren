"""
Dataset loading and management utilities
"""

import json
import random
from typing import List, Dict, Optional
from datasets import load_dataset
from huggingface_hub import HfApi


def normalize_item(x: dict, dataset_name: str) -> Optional[Dict[str, str]]:
    """Normalize dataset items to a common format."""
    # GSM8k-lite
    if dataset_name == "openai/gsm8k-lite":
        q = x.get("question", "").strip()
        a = x.get("answer", "").strip()
        if q:
            return {"prompt": q, "reference": a}

    # ARC-Easy
    if dataset_name == "allenai/ARC-Easy":
        q = x.get("question", "").strip()
        if q and "choices" in x and "answerKey" in x:
            for lbl, txt in zip(x["choices"]["label"], x["choices"]["text"]):
                if lbl == x["answerKey"]:
                    return {"prompt": q, "reference": txt}

    # SciQ
    if dataset_name == "sciq":
        q = x.get("question", "").strip()
        a = x.get("correct_answer", "").strip()
        if q and a:
            return {"prompt": q, "reference": a}

    # TruthfulQA
    if dataset_name == "truthful_qa":
        q = x.get("question", "").strip()
        a = x.get("best_answer", "").strip()
        if q and a:
            return {"prompt": q, "reference": a}

    # OpenBookQA
    if dataset_name == "openbookqa":
        q = x["question_stem"].strip()
        if q and "choices" in x:
            for lbl, txt in zip(x["choices"]["label"], x["choices"]["text"]):
                if lbl == x["answerKey"]:
                    return {"prompt": q, "reference": txt}

    # TriviaQA
    if dataset_name == "trivia_qa":
        q = x.get("question", "").strip()
        if q:
            ans = x.get("answer", {})
            if isinstance(ans, dict):
                a = ans.get("value", "")
            else:
                a = ""
            a = (a or "").strip()
            return {"prompt": q, "reference": a}

    return None


def safe_sample(dataset_name: str, subset: Optional[str], split: str, k: int) -> List[Dict[str, str]]:
    """Safely load and sample from a dataset."""
    try:
        ds = load_dataset(dataset_name, subset, split=split)

        def convert(x):
            return normalize_item(x, dataset_name)

        ds2 = ds.map(
            convert,
            num_proc=4,
            remove_columns=ds.column_names,
        )

        ds3 = [x for x in ds2 if x and x["prompt"] and x["reference"]]
        random.shuffle(ds3)
        return ds3[:k]

    except Exception as e:
        print(f"Dataset {dataset_name} failed to load: {e}")
        return []


def load_eval_dataset(n: int = 120) -> List[Dict[str, str]]:
    """
    Load a diverse evaluation dataset from multiple sources.
    
    Args:
        n: Total number of samples to load
        
    Returns:
        List of dicts with 'prompt' and 'reference' keys
    """
    print(f"Loading evaluation dataset ({n} samples)...")
    
    parts = []
    parts += safe_sample("openai/gsm8k-lite", None, "train", 40)
    parts += safe_sample("allenai/ARC-Easy", None, "train", 40)
    parts += safe_sample("sciq", None, "train", 20)
    parts += safe_sample("truthful_qa", "generation", "validation", 20)
    parts += safe_sample("openbookqa", "main", "train", 20)
    parts += safe_sample("trivia_qa", "unfiltered", "validation", 20)

    final = [x for x in parts if x["prompt"].strip() and x["reference"].strip()]
    random.shuffle(final)
    
    result = final[:n]
    print(f"Loaded {len(result)} evaluation items")
    return result


def upload_dataset_to_hub(eval_data: List[Dict[str, str]], repo_id: str, token: Optional[str] = None):
    """
    Upload evaluation dataset to HuggingFace Hub.
    
    Args:
        eval_data: List of evaluation samples
        repo_id: HuggingFace repo ID (e.g. "username/dataset-name")
        token: Optional HF token (will use cached if not provided)
    """
    eval_dataset_path = "eval_data.jsonl"
    
    with open(eval_dataset_path, "w", encoding="utf-8") as f:
        for row in eval_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    print(f"Saved dataset to {eval_dataset_path}")
    
    api = HfApi(token=token)
    
    print(f"Uploading to HF Hub: {repo_id}")
    api.upload_file(
        path_or_fileobj=eval_dataset_path,
        path_in_repo="data/data.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    print("Upload complete!")


def load_dataset_from_hub(repo_id: str) -> List[Dict[str, str]]:
    """
    Load evaluation dataset from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repo ID
        
    Returns:
        List of evaluation samples
    """
    print(f"Loading dataset from HF Hub: {repo_id}")
    
    loaded_eval = load_dataset(repo_id, split="train")
    loaded_eval = [
        {"prompt": x["prompt"], "reference": x["reference"]}
        for x in loaded_eval
    ]
    
    print(f"Loaded {len(loaded_eval)} items")
    return loaded_eval


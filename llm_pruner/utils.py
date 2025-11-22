"""
Utility functions
"""

import re
import torch


def count_parameters(model) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def format_prompt(q: str) -> str:
    """Format a question into a chat prompt."""
    return "<|im_start|>user\n" + q.strip() + "<|im_end|>\n<|im_start|>assistant\n<think>"


def clean_output(text: str) -> str:
    """Clean model output by removing special tokens and tags."""
    if "<think>" in text:
        text = text.split("<think>")[-1]
    text = re.sub(r".*?</(think|analysis|reasoning)>", "", text, flags=re.DOTALL)
    return text.replace("<|im_end|>", "").strip()


def get_device() -> str:
    """Get the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


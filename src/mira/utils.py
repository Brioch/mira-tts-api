import re
import gc
import torch


def split_text(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return sentences


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

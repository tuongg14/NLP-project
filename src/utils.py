import random
import os
import torch
import numpy as np
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(state: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))

def load_checkpoint(path: str, device="cpu"):
    return torch.load(path, map_location=device)
# src/config_loader.py
import yaml
from pathlib import Path

def load_config(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # Basic defaults (một số sanity checks)
    cfg.setdefault("seed", 42)
    cfg.setdefault("data", {})
    cfg.setdefault("vocab", {"max_size": 10000, "specials": ["<unk>", "<pad>", "<sos>", "<eos>"]})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("paths", {})
    return cfg

# small helper to get path join
def get_path(config, key):
    base = Path(config.get("paths", {}).get("checkpoint_dir", "./checkpoints"))
    return base / config.get("paths", {}).get(key, "")

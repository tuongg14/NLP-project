from pathlib import Path
import yaml

def load_config(path=None):
    # project root = NLP-project
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    if path is None:
        path = PROJECT_ROOT / "config" / "config.yml"

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

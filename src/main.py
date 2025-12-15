import os
import sys
from pathlib import Path
from train import main
from evaluate import run_evaluate
from infer import translate

# Lấy đường dẫn project root (NLP-project)
PROJECT_ROOT = Path.cwd().parent

# Chuyển working directory về project root
os.chdir(PROJECT_ROOT)

# Thêm src vào PYTHONPATH để import được các file .py
sys.path.append(str(PROJECT_ROOT / "src"))

print("Working directory:", os.getcwd())
print("src in path:", PROJECT_ROOT / "src")

# TRAIN 
#main()
# EVALUATE
run_evaluate()
# INFER
translate("A man is riding a bike")
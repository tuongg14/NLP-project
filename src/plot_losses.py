import json
from pathlib import Path
import matplotlib.pyplot as plt

# ================== LOAD METRICS ==================
results_dir = Path("../results")
metrics_path = results_dir / "metrics.json"

if not metrics_path.exists():
    raise FileNotFoundError(
        f"Không tìm thấy {metrics_path}. "
        "Hãy chạy train.py trước để sinh metrics.json"
    )

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

train_loss = metrics["train_loss"]
val_loss = metrics["val_loss"]

epochs = list(range(1, len(train_loss) + 1))

# ================== PLOT ==================
plt.figure(figsize=(8, 5))

# Train Loss (đường xanh)
plt.plot(
    epochs,
    train_loss,
    marker="o",
    linewidth=2,
    label="Train Loss",
)

# Val Loss (đường đỏ)
plt.plot(
    epochs,
    val_loss,
    marker="s",
    linewidth=2,
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss vs Validation Loss")
plt.legend()
plt.grid(True)

# ================== SAVE ==================
save_path = results_dir / "loss_curve.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print("Đã lưu biểu đồ tại:", save_path)

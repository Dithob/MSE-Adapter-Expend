import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. 输出目录
# =========================
save_dir = Path("./training_figures_17epoch")
save_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 17个epoch的真实实验数据
# =========================
epochs = list(range(1, 18))

train_loss = [
    3.1087, 0.8499, 0.7469, 0.6986, 0.6247, 0.5836, 0.5704, 0.5650, 0.5537,
    0.5478, 0.5408, 0.5368, 0.5294, 0.5255, 0.5206, 0.5138, 0.5100
]

val_acc = [
    0.4193, 0.4256, 0.4373, 0.5230, 0.5681, 0.5825, 0.5870, 0.5924, 0.5906,
    0.5870, 0.6096, 0.6123, 0.6105, 0.6222, 0.6087, 0.6114, 0.6087
]

val_weighted_f1 = [
    0.2686, 0.2710, 0.3168, 0.4723, 0.5488, 0.5442, 0.5362, 0.5739, 0.5407,
    0.5369, 0.5706, 0.5833, 0.5685, 0.5980, 0.5703, 0.5783, 0.5813
]

df = pd.DataFrame({
    "epoch": epochs,
    "train_loss": train_loss,
    "val_acc": val_acc,
    "val_weighted_f1": val_weighted_f1
})

# =========================
# 3. 关键统计
# =========================
best_f1_idx = df["val_weighted_f1"].idxmax()
best_acc_idx = df["val_acc"].idxmax()

best_f1_epoch = int(df.loc[best_f1_idx, "epoch"])
best_f1_value = float(df.loc[best_f1_idx, "val_weighted_f1"])

best_acc_epoch = int(df.loc[best_acc_idx, "epoch"])
best_acc_value = float(df.loc[best_acc_idx, "val_acc"])

print(f"Best Val Weighted F1: {best_f1_value:.4f} at epoch {best_f1_epoch}")
print(f"Best Val Acc: {best_acc_value:.4f} at epoch {best_acc_epoch}")

# =========================
# 4. 全局绘图风格
# =========================
plt.rcParams["figure.figsize"] = (9, 5.2)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["font.size"] = 11

# 如果需要中文显示，取消下面两行注释
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False

# =========================
# 5. 图1：训练损失曲线
# =========================
plt.figure()
plt.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, label="Train Loss")
plt.axvline(best_f1_epoch, linestyle="--", label=f"Best Val F1 Epoch ({best_f1_epoch})")

plt.annotate(
    f"Best checkpoint epoch = {best_f1_epoch}",
    xy=(best_f1_epoch, df.loc[df['epoch'] == best_f1_epoch, 'train_loss'].iloc[0]),
    xytext=(11.5, 1.2),
    arrowprops=dict(arrowstyle="->")
)

plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.xlim(1, 17)
plt.legend()
plt.tight_layout()
plt.savefig(save_dir / "01_training_loss_curve.png", dpi=300, bbox_inches="tight")
plt.savefig(save_dir / "01_training_loss_curve.pdf", bbox_inches="tight")
plt.show()

# =========================
# 6. 图2：验证集指标曲线
# =========================
plt.figure()
plt.plot(df["epoch"], df["val_acc"], marker="o", linewidth=2, label="Validation Accuracy")
plt.plot(df["epoch"], df["val_weighted_f1"], marker="s", linewidth=2, label="Validation Weighted F1")

plt.axvline(best_f1_epoch, linestyle="--")

plt.scatter([best_f1_epoch], [best_f1_value], s=80)
plt.annotate(
    f"Best F1 = {best_f1_value:.4f}\n(epoch {best_f1_epoch})",
    xy=(best_f1_epoch, best_f1_value),
    xytext=(14.2, 0.52),
    arrowprops=dict(arrowstyle="->")
)

plt.scatter([best_acc_epoch], [best_acc_value], s=80)
plt.annotate(
    f"Best Acc = {best_acc_value:.4f}\n(epoch {best_acc_epoch})",
    xy=(best_acc_epoch, best_acc_value),
    xytext=(10.5, 0.62),
    arrowprops=dict(arrowstyle="->")
)

plt.title("Validation Metrics Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.xlim(1, 17)
plt.ylim(0.2, 0.68)
plt.legend()
plt.tight_layout()
plt.savefig(save_dir / "02_validation_metrics_curve.png", dpi=300, bbox_inches="tight")
plt.savefig(save_dir / "02_validation_metrics_curve.pdf", bbox_inches="tight")
plt.show()

# =========================
# 7. 图3：checkpoint总结图
# =========================
checkpoint_epochs = [1, 2, 3, 4, 5, 8, 12, 14]
ckpt_df = df[df["epoch"].isin(checkpoint_epochs)]

plt.figure()
plt.plot(df["epoch"], df["val_weighted_f1"], marker="o", linewidth=2, label="Validation Weighted F1")
plt.scatter(ckpt_df["epoch"], ckpt_df["val_weighted_f1"], s=85, label="Saved Checkpoints")
plt.scatter([best_f1_epoch], [best_f1_value], s=120, marker="*", label="Best Checkpoint")

plt.annotate(
    f"Best checkpoint\nEpoch {best_f1_epoch}, F1 = {best_f1_value:.4f}",
    xy=(best_f1_epoch, best_f1_value),
    xytext=(10.5, 0.42),
    arrowprops=dict(arrowstyle="->")
)

plt.title("Checkpoint Selection Summary")
plt.xlabel("Epoch")
plt.ylabel("Weighted F1")
plt.xlim(1, 17)
plt.ylim(0.24, 0.64)
plt.legend()
plt.tight_layout()
plt.savefig(save_dir / "03_checkpoint_summary.png", dpi=300, bbox_inches="tight")
plt.savefig(save_dir / "03_checkpoint_summary.pdf", bbox_inches="tight")
plt.show()

# =========================
# 8. 保存数据
# =========================
df.to_csv(save_dir / "training_metrics_17epoch.csv", index=False, encoding="utf-8-sig")

print(f"All figures and csv files have been saved to: {save_dir.resolve()}")
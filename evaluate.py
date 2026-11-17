
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from pathlib import Path
from dataset import val_dataset
from model import UNet


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model
model = UNet().to(DEVICE)

checkpoint = torch.load(
    OUTPUT_DIR / "best_unet.pth",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])


# Metric calculation
results = {
    "Cytoplasm": {"dice": [], "iou": [], "precision": [], "recall": []},
    "Nucleus": {"dice": [], "iou": [], "precision": [], "recall": []}
}

names = ["Cytoplasm", "Nucleus"]

with torch.no_grad():

    for image, target in val_dataset:

        image = image.unsqueeze(0).to(DEVICE)
        target = target.unsqueeze(0).to(DEVICE)

        prediction = (torch.sigmoid(model(image)) >= 0.5).float()

        for channel, name in enumerate(names):

            pred = prediction[0, channel].bool()
            true = target[0, channel] >= 0.5

            tp = (pred & true).sum().item()
            fp = (pred & ~true).sum().item()
            fn = (~pred & true).sum().item()

            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            results[name]["dice"].append(dice)
            results[name]["iou"].append(iou)
            results[name]["precision"].append(precision)
            results[name]["recall"].append(recall)


# Average metrics across validation images
summary = {}

print("\n========== VALIDATION RESULTS ==========")

for name in names:

    summary[name] = {}

    print(f"\n{name}")

    for metric, values in results[name].items():

        average = float(np.mean(values))
        summary[name][metric] = average

        print(f"{metric.capitalize()}: {average:.4f} ({average*100:.2f}%)")


summary["Mean Dice"] = float(np.mean([
    summary["Cytoplasm"]["dice"],
    summary["Nucleus"]["dice"]
]))

print("\nMean Dice:", round(summary["Mean Dice"], 4))


# Save metrics
with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
    json.dump(summary, f, indent=4)


# Plot metric comparison
metrics = ["dice", "iou", "precision", "recall"]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    [summary["Cytoplasm"][m] for m in metrics],
    width,
    label="Cytoplasm"
)

plt.bar(
    x + width / 2,
    [summary["Nucleus"][m] for m in metrics],
    width,
    label="Nucleus"
)

plt.xticks(x, [m.capitalize() for m in metrics])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Segmentation Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "metric_comparison.png", dpi=200)
plt.close()


# Plot training history
with open(OUTPUT_DIR / "training_history.json", "r") as f:
    history = json.load(f)

epochs = [item["epoch"] for item in history]

plt.figure(figsize=(9, 5))
plt.plot(epochs, [item["train_loss"] for item in history], label="Training Loss")
plt.plot(epochs, [item["val_loss"] for item in history], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=200)
plt.close()


# Plot Dice progression
plt.figure(figsize=(9, 5))

plt.plot(epochs, [item["cytoplasm_dice"] for item in history],
         label="Cytoplasm Dice")

plt.plot(epochs, [item["nucleus_dice"] for item in history],
         label="Nucleus Dice")

plt.plot(epochs, [item["mean_dice"] for item in history],
         label="Mean Dice", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Validation Dice Progression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dice_curve.png", dpi=200)
plt.close()


print("\nAll evaluation files generated in outputs/")


import os
import json
import torch
import torch.nn as nn

from pathlib import Path
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from dataset import train_loader, val_loader
from model import UNet


# ==========================================
# 1. Configuration
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 30
LEARNING_RATE = 0.0001

MODEL_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_unet.pth"
HISTORY_PATH = MODEL_DIR / "training_history.json"

USE_AMP = DEVICE.type == "cuda"

print("Device:", DEVICE)
print("Mixed precision:", USE_AMP)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)


# ==========================================
# 2. Initialize model
# ==========================================

model = UNet().to(DEVICE)

criterion_bce = nn.BCEWithLogitsLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scaler = GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ==========================================
# 3. Dice Loss
# ==========================================

def dice_loss(predictions, targets, smooth=1e-6):

    probabilities = torch.sigmoid(predictions)

    probabilities = probabilities.flatten(2)
    targets = targets.flatten(2)

    intersection = (probabilities * targets).sum(dim=2)

    denominator = (
        probabilities.sum(dim=2)
        + targets.sum(dim=2)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice.mean()


# ==========================================
# 4. Combined Loss
# ==========================================

def combined_loss(predictions, targets):

    bce = criterion_bce(predictions, targets)

    dice = dice_loss(predictions, targets)

    return bce + dice


# ==========================================
# 5. Evaluation Metrics
# ==========================================

def calculate_metrics(predictions, targets, threshold=0.5):

    probabilities = torch.sigmoid(predictions)

    predicted_masks = probabilities >= threshold
    true_masks = targets >= 0.5

    predicted_masks = predicted_masks.float()
    true_masks = true_masks.float()

    # Calculate separately for each output channel
    dimensions = (0, 2, 3)

    intersection = (
        predicted_masks * true_masks
    ).sum(dim=dimensions)

    predicted_sum = predicted_masks.sum(dim=dimensions)
    target_sum = true_masks.sum(dim=dimensions)

    union = predicted_sum + target_sum - intersection

    dice = (
        2 * intersection + 1e-6
    ) / (
        predicted_sum + target_sum + 1e-6
    )

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    return dice, iou


# ==========================================
# 6. Training Function
# ==========================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):

            predictions = model(images)

            loss = combined_loss(predictions, masks)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ==========================================
# 7. Validation Function
# ==========================================

def validate():

    model.eval()

    total_loss = 0.0

    total_dice = torch.zeros(2, device=DEVICE)
    total_iou = torch.zeros(2, device=DEVICE)

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            with autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP
            ):

                predictions = model(images)

                loss = combined_loss(predictions, masks)

            dice, iou = calculate_metrics(
                predictions.float(),
                masks.float()
            )

            total_loss += loss.item()

            total_dice += dice
            total_iou += iou

    average_loss = total_loss / len(val_loader)

    average_dice = total_dice / len(val_loader)
    average_iou = total_iou / len(val_loader)

    return average_loss, average_dice, average_iou


# ==========================================
# 8. Main Training Loop
# ==========================================

if __name__ == "__main__":

    history = []

    best_dice = 0.0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch()

        val_loss, val_dice, val_iou = validate()

        cytoplasm_dice = val_dice[0].item()
        nucleus_dice = val_dice[1].item()

        cytoplasm_iou = val_iou[0].item()
        nucleus_iou = val_iou[1].item()

        mean_dice = (
            cytoplasm_dice + nucleus_dice
        ) / 2

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        print(
            f"Cytoplasm Dice: {cytoplasm_dice:.4f}"
        )

        print(
            f"Nucleus Dice: {nucleus_dice:.4f}"
        )

        print(
            f"Cytoplasm IoU: {cytoplasm_iou:.4f}"
        )

        print(
            f"Nucleus IoU: {nucleus_iou:.4f}"
        )

        print(f"Mean Dice: {mean_dice:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "cytoplasm_dice": cytoplasm_dice,
            "nucleus_dice": nucleus_dice,
            "cytoplasm_iou": cytoplasm_iou,
            "nucleus_iou": nucleus_iou,
            "mean_dice": mean_dice
        })

        # Save best model
        if mean_dice > best_dice:

            best_dice = mean_dice

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_mean_dice": best_dice
                },
                BEST_MODEL_PATH
            )

            print("Best model saved!")

        # Save training history
        with open(HISTORY_PATH, "w") as file:

            json.dump(history, file, indent=4)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\nTraining completed!")

    print(f"Best validation Mean Dice: {best_dice:.4f}")

    print("Best model:", BEST_MODEL_PATH)
    print("Training history:", HISTORY_PATH)

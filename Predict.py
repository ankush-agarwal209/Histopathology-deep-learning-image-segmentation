
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dataset import val_dataset
from model import UNet


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load model
model = UNet().to(device)

checkpoint = torch.load(
    "outputs/best_unet.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")
print("Best epoch:", checkpoint["epoch"])


# Output folder
output_dir = Path("outputs/predictions")
output_dir.mkdir(parents=True, exist_ok=True)


# Generate predictions
with torch.no_grad():

    for index in range(min(5, len(val_dataset))):

        image, mask = val_dataset[index]

        input_tensor = image.unsqueeze(0).to(device)

        logits = model(input_tensor)

        probabilities = torch.sigmoid(logits)

        prediction = (
            probabilities > 0.5
        ).float()

        # Convert tensors to NumPy
        original = image.permute(1, 2, 0).numpy()

        true_cytoplasm = mask[0].numpy()
        true_nucleus = mask[1].numpy()

        pred_cytoplasm = prediction[0, 0].cpu().numpy()
        pred_nucleus = prediction[0, 1].cpu().numpy()

        # Plot results
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original Image")

        axes[0, 1].imshow(true_cytoplasm, cmap="gray")
        axes[0, 1].set_title("Actual Cytoplasm")

        axes[0, 2].imshow(pred_cytoplasm, cmap="gray")
        axes[0, 2].set_title("Predicted Cytoplasm")

        axes[1, 0].imshow(original)
        axes[1, 0].set_title("Original Image")

        axes[1, 1].imshow(true_nucleus, cmap="gray")
        axes[1, 1].set_title("Actual Nucleus")

        axes[1, 2].imshow(pred_nucleus, cmap="gray")
        axes[1, 2].set_title("Predicted Nucleus")

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            output_dir / f"prediction_{index + 1}.png",
            dpi=200
        )

        plt.close(fig)

        print(f"Prediction {index + 1} saved.")

print("\nAll predictions completed!")

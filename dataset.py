
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np


# Dataset location
dataset_path = Path(
    r"C:\Users\Ankush Agarwal\Downloads\archive\Cx22_Multi_Train"
)


# Dataset folders
image_dir = dataset_path / "Train" / "images"

cytoplasm_dir = (
    dataset_path / "Train" / "gt_mask"
    / "Cytoplasm" / "Sem_seg"
)

nucleus_dir = (
    dataset_path / "Train" / "gt_mask"
    / "Nucleus" / "Sem_seg"
)


# Dataset class
class Cx22Dataset(Dataset):

    def __init__(self, image_dir, cytoplasm_dir, nucleus_dir):

        self.image_dir = image_dir
        self.cytoplasm_dir = cytoplasm_dir
        self.nucleus_dir = nucleus_dir

        self.image_files = sorted(image_dir.glob("*.png"))

        # Check corresponding masks exist
        for image_path in self.image_files:

            cytoplasm_path = cytoplasm_dir / image_path.name
            nucleus_path = nucleus_dir / image_path.name

            if not cytoplasm_path.exists():
                raise FileNotFoundError(
                    f"Cytoplasm mask missing: {cytoplasm_path}"
                )

            if not nucleus_path.exists():
                raise FileNotFoundError(
                    f"Nucleus mask missing: {nucleus_path}"
                )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_path = self.image_files[index]

        cytoplasm_path = self.cytoplasm_dir / image_path.name
        nucleus_path = self.nucleus_dir / image_path.name

        # Load original image
        image = Image.open(image_path).convert("RGB")

        # Load both ground-truth masks
        cytoplasm = Image.open(cytoplasm_path).convert("L")
        nucleus = Image.open(nucleus_path).convert("L")

        # Convert images into NumPy arrays
        image = np.array(image, dtype=np.float32) / 255.0

        cytoplasm = np.array(cytoplasm, dtype=np.float32) / 255.0
        nucleus = np.array(nucleus, dtype=np.float32) / 255.0

        # Convert NumPy arrays into PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)

        cytoplasm = torch.from_numpy(cytoplasm)
        nucleus = torch.from_numpy(nucleus)

        # Combine two masks into two output channels
        mask = torch.stack([cytoplasm, nucleus], dim=0)

        return image, mask


# Create dataset
dataset = Cx22Dataset(
    image_dir,
    cytoplasm_dir,
    nucleus_dir
)


# Split dataset: 80% training, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# Test dataset and DataLoader
if __name__ == "__main__":

    print("Total images:", len(dataset))
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    images, masks = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch mask shape:", masks.shape)

    print("Cytoplasm mask shape:", masks[:, 0].shape)
    print("Nucleus mask shape:", masks[:, 1].shape)

"""Project paths and configuration."""

import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_unet.pth"

# Set CX22_DATASET_DIR in your environment if the dataset is stored elsewhere.
DATASET_ROOT = Path(
    os.getenv(
        "CX22_DATASET_DIR",
        r"C:\Users\Ankush Agarwal\Downloads\archive\Cx22_Multi_Train",
    )
)

TEST_IMAGE_DIR = DATASET_ROOT / "Test" / "images"
TEST_CYTOPLASM_DIR = DATASET_ROOT / "Test" / "gt_mask" / "Cytoplasm" / "Sem_seg"
TEST_NUCLEUS_DIR = DATASET_ROOT / "Test" / "gt_mask" / "Nucleus" / "Sem_seg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (512, 512)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Published group mean and SD for N:C ratio, Mishra et al. (2023), Table 1.
# Exploratory reference values only; they are not diagnostic thresholds.
REFERENCE_NC = {
    "NILM": (0.03, 0.01),
    "ASC-US": (0.24, 0.09),
    "LSIL": (0.27, 0.05),
    "ASC-H": (0.77, 0.13),
    "HSIL": (0.74, 0.13),
    "SCC": (0.80, 0.12),
}

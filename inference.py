"""Model loading, image preprocessing, and segmentation inference."""

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import streamlit as st

from config import DEVICE, IMAGE_SIZE, MODEL_PATH
from model import UNet


@st.cache_resource
def load_model():
    """Load the trained U-Net checkpoint once per Streamlit session."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}. Place best_unet.pth in outputs/."
        )

    model = UNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()
    return model


def preprocess_image(image):
    resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE), resized


def predict_masks(model, image, threshold=0.5):
    tensor, resized = preprocess_image(image)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor)).squeeze(0).cpu().numpy()
    cytoplasm = (probabilities[0] >= threshold).astype(np.uint8)
    nucleus = (probabilities[1] >= threshold).astype(np.uint8)
    return resized, probabilities, cytoplasm, nucleus


def create_overlay(image, mask, color):
    base = np.asarray(image).astype(np.float32)
    selected = np.asarray(mask) > 0
    base[selected] = 0.5 * base[selected] + 0.5 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def image_to_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

"""Ground-truth loading and segmentation evaluation metrics."""

from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    IMAGE_SIZE,
    SUPPORTED_EXTENSIONS,
    TEST_CYTOPLASM_DIR,
    TEST_IMAGE_DIR,
    TEST_NUCLEUS_DIR,
)
from inference import predict_masks
from morphology import calculate_morphology


def load_mask(path):
    mask = Image.open(path).convert("L").resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    return (np.asarray(mask, dtype=np.uint8) > 127).astype(np.uint8)


def calculate_metrics(prediction, ground_truth):
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(ground_truth, dtype=bool)
    tp = int(np.logical_and(pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "Dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "IoU": tp / (tp + fp + fn + eps),
        "Precision": precision,
        "Recall": recall,
        "F1-score": (2 * precision * recall) / (precision + recall + eps),
        "Specificity": tn / (tn + fp + eps),
        "Pixel accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def find_test_pairs():
    folders = [TEST_IMAGE_DIR, TEST_CYTOPLASM_DIR, TEST_NUCLEUS_DIR]
    missing = [str(folder) for folder in folders if not folder.is_dir()]
    if missing:
        raise FileNotFoundError("Missing Test folders:\n" + "\n".join(missing))

    image_files = sorted(
        path for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    pairs, unmatched = [], []
    for image_path in image_files:
        cyto_path = TEST_CYTOPLASM_DIR / image_path.name
        nucleus_path = TEST_NUCLEUS_DIR / image_path.name
        if cyto_path.is_file() and nucleus_path.is_file():
            pairs.append((image_path, cyto_path, nucleus_path))
        else:
            unmatched.append(image_path.name)
    return pairs, unmatched


def evaluate_image(model, image_path, cyto_path, nucleus_path):
    original = Image.open(image_path).convert("RGB")
    resized, probabilities, pred_cyto, pred_nucleus = predict_masks(model, original)
    gt_cyto = load_mask(cyto_path)
    gt_nucleus = load_mask(nucleus_path)
    return {
        "image": resized,
        "pred_cyto": pred_cyto,
        "pred_nucleus": pred_nucleus,
        "gt_cyto": gt_cyto,
        "gt_nucleus": gt_nucleus,
        "cyto_metrics": calculate_metrics(pred_cyto, gt_cyto),
        "nucleus_metrics": calculate_metrics(pred_nucleus, gt_nucleus),
        "morphology": calculate_morphology(pred_cyto, pred_nucleus, probabilities),
        "probabilities": probabilities,
    }


def evaluate_full_test_set(model, pairs, progress_callback=None):
    per_class = {"Cytoplasm": [], "Nucleus": []}
    for index, pair in enumerate(pairs):
        result = evaluate_image(model, *pair)
        per_class["Cytoplasm"].append(result["cyto_metrics"])
        per_class["Nucleus"].append(result["nucleus_metrics"])
        if progress_callback:
            progress_callback((index + 1) / len(pairs), index + 1, len(pairs))

    return {
        label: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in rows[0]
        }
        for label, rows in per_class.items()
    }

"""Image-level morphology measurements and research-reference comparison."""

import cv2
import numpy as np

from config import REFERENCE_NC


def binary_shape_measurements(mask):
    """Measure the aggregate foreground mask, not individual cell instances."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    area = int(binary.sum())
    total_pixels = int(binary.size)
    area_percentage = area / total_pixels * 100.0 if total_pixels else 0.0

    if area == 0:
        return {
            "area_pixels": 0,
            "area_percentage": 0.0,
            "perimeter_pixels": 0.0,
            "equivalent_diameter_pixels": 0.0,
            "circularity": 0.0,
            "compactness": 0.0,
            "eccentricity": 0.0,
            "solidity": 0.0,
            "connected_regions": 0,
        }

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    equivalent_diameter = float(np.sqrt(4.0 * area / np.pi))
    circularity = float(4.0 * np.pi * area / perimeter**2) if perimeter > 0 else 0.0
    compactness = float(perimeter**2 / (4.0 * np.pi * area))
    hull_area = float(sum(cv2.contourArea(cv2.convexHull(contour)) for contour in contours))
    solidity = float(min(area / hull_area, 1.0)) if hull_area > 0 else 0.0

    ys, xs = np.nonzero(binary)
    if len(xs) > 1:
        covariance = np.cov(np.vstack((xs, ys)), bias=True)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        major, minor = float(eigenvalues[-1]), float(eigenvalues[0])
        eccentricity = float(np.sqrt(max(0.0, 1.0 - minor / major))) if major > 0 else 0.0
    else:
        eccentricity = 0.0

    return {
        "area_pixels": area,
        "area_percentage": area_percentage,
        "perimeter_pixels": perimeter,
        "equivalent_diameter_pixels": equivalent_diameter,
        "circularity": circularity,
        "compactness": compactness,
        "eccentricity": eccentricity,
        "solidity": solidity,
        "connected_regions": len(contours),
    }


def calculate_morphology(cytoplasm_mask, nucleus_mask, probabilities=None):
    cytoplasm = (np.asarray(cytoplasm_mask) > 0).astype(np.uint8)
    nucleus = (np.asarray(nucleus_mask) > 0).astype(np.uint8)
    cyto_area = int(cytoplasm.sum())
    nucleus_area = int(nucleus.sum())
    nc_ratio = float(nucleus_area / cyto_area) if cyto_area > 0 else 0.0
    cyto_shape = binary_shape_measurements(cytoplasm)
    nucleus_shape = binary_shape_measurements(nucleus)

    result = {
        "cytoplasm_area": cyto_area,
        "nucleus_area": nucleus_area,
        "nc_ratio": nc_ratio,
        "cyto_shape": cyto_shape,
        "nucleus_shape": nucleus_shape,
        "cytoplasm_percentage": cyto_shape["area_percentage"],
        "nucleus_percentage": nucleus_shape["area_percentage"],
        "total_segmented_percentage": float(np.logical_or(cytoplasm, nucleus).mean() * 100.0),
    }
    if probabilities is not None:
        probs = np.asarray(probabilities)
        result["cytoplasm_mean_probability"] = float(probs[0].mean() * 100.0)
        result["nucleus_mean_probability"] = float(probs[1].mean() * 100.0)
    return result


def reference_similarity(nc_ratio):
    """Return normalized Gaussian similarity scores, not diagnostic probabilities."""
    likelihoods = {}
    for category, (mean, sd) in REFERENCE_NC.items():
        z_score = (nc_ratio - mean) / sd
        likelihoods[category] = float(np.exp(-0.5 * z_score * z_score))
    total = sum(likelihoods.values())
    if total <= 0 or not np.isfinite(total):
        return {category: 0.0 for category in likelihoods}
    return {category: value / total * 100.0 for category, value in likelihoods.items()}

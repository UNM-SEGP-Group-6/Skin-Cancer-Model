# lesion_processing.py
# ============================================================
# Lesion Processing Module 
# ============================================================

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════
# INPUT HANDLING
# ══════════════════════════════════════════════════════════════

def preprocess_mask(image: np.ndarray, user_input):
    """
    Accepts:
        - binary mask (same size as image)
        - OR bounding box (x, y, w, h)

    Returns:
        binary mask (uint8, 0 or 255)
    """
    h, w = image.shape[:2]

    # Case 1: Already a mask
    if isinstance(user_input, np.ndarray):
        mask = user_input.copy()
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8) * 255

    # Case 2: Bounding box
    elif isinstance(user_input, (list, tuple)) and len(user_input) == 4:
        x, y, bw, bh = user_input
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y:y+bh, x:x+bw] = 255
        return mask

    else:
        raise ValueError("Invalid user_input: must be mask or bounding box")


# ══════════════════════════════════════════════════════════════
# SEGMENTATION (GRABCUT — MATCHES YOUR PIPELINE STYLE)
# ══════════════════════════════════════════════════════════════

def refine_mask_grabcut(image: np.ndarray, init_mask: np.ndarray, iter_count=5, fast_mode=False):
    """
    Refines user mask using GrabCut.
    Compatible with OpenCV pipeline style.
    """
    if fast_mode:
        return init_mask
    
    mask = np.where(init_mask > 0, cv2.GC_FGD, cv2.GC_BGD).astype("uint8")

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, None, bg_model, fg_model, iter_count, cv2.GC_INIT_WITH_MASK)

    refined = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    return refined


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def compute_area(mask: np.ndarray):
    return int(np.sum(mask > 0))


def compute_perimeter(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(sum(cv2.arcLength(cnt, True) for cnt in contours))


def compute_circularity(area, perimeter):
    if perimeter == 0:
        return 0.0
    return float((4 * np.pi * area) / (perimeter ** 2))


def compute_asymmetry(mask: np.ndarray):
    """
    Horizontal symmetry comparison
    """
    flipped = np.fliplr(mask)
    diff = np.abs(mask.astype(np.int32) - flipped.astype(np.int32))

    total = np.sum(mask > 0)
    if total == 0:
        return 0.0

    return float(np.sum(diff > 0) / total)


def compute_color_variance(image: np.ndarray, mask: np.ndarray):
    """
    Uses BGR image (same as pipeline)
    """
    pixels = image[mask > 0]

    if len(pixels) == 0:
        return 0.0

    return float(np.var(pixels, axis=0).mean())


# ══════════════════════════════════════════════════════════════
# MAIN FEATURE PIPELINE
# ══════════════════════════════════════════════════════════════

def extract_features(image: np.ndarray, user_input):
    """
    Main entry point.

    Args:
        image: BGR image (AFTER hair removal)
        user_input: mask or bounding box

    Returns:
        dict of lesion features
    """

    # Step 1 — preprocess input
    init_mask = preprocess_mask(image, user_input)

    # Step 2 — refine segmentation
    refined_mask = refine_mask_grabcut(image, init_mask, fast_mode=True)

    # Step 3 — compute features
    area = compute_area(refined_mask)
    perimeter = compute_perimeter(refined_mask)
    circularity = compute_circularity(area, perimeter)
    asymmetry = compute_asymmetry(refined_mask)
    color_variance = compute_color_variance(image, refined_mask)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "asymmetry": asymmetry,
        "color_variance": color_variance,
    }


# ══════════════════════════════════════════════════════════════
# GROWTH COMPARISON (FOR YOUR REPORT CLAIM)
# ══════════════════════════════════════════════════════════════

def compare_growth(f1: dict, f2: dict):
    return {
        "area_change": f2["area"] - f1["area"],
        "circularity_change": f2["circularity"] - f1["circularity"],
        "asymmetry_change": f2["asymmetry"] - f1["asymmetry"],
        "color_change": f2["color_variance"] - f1["color_variance"],
    }
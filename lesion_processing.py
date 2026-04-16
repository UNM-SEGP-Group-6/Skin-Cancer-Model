# lesion_processing.py
# ============================================================
# Lesion Processing Module 
# ============================================================

import cv2
import numpy as np
from abc import ABC, abstractmethod

# ══════════════════════════════════════════════════════════════
# SEGMENTATION STRATEGIES
# ══════════════════════════════════════════════════════════════

class SegmentationStrategy(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray) -> tuple:
        """Returns a bounding box tuple (x, y, w, h) or None."""
        pass

class OtsuSegmentation(SegmentationStrategy):
    def segment(self, image: np.ndarray) -> tuple:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Heavy blur to remove skin texture
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Sort by area descending to find the largest valid contour
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for c in contours:
            area = cv2.contourArea(c)
            if 500 < area < (h * w * 0.7):
                return cv2.boundingRect(c)
                
        return None


def refine_mask_grabcut(image: np.ndarray, user_input, iter_count=5):
    """
    Refines segmentation using GrabCut.
    Handles both bounding boxes (tuple) and probability masks (ndarray).
    """
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    
    # Initialize the mask that GrabCut will modify
    mask = np.zeros(image.shape[:2], np.uint8)

    if isinstance(user_input, (list, tuple)) and len(user_input) == 4:
        # User provided a bounding box
        rect = tuple(user_input)
        cv2.grabCut(image, mask, rect, bg_model, fg_model, iter_count, cv2.GC_INIT_WITH_RECT)
    elif isinstance(user_input, np.ndarray):
        # User provided a mask (assuming it's formatted with GC_BGD, GC_PR_FGD, etc.)
        mask = user_input.copy()
        cv2.grabCut(image, mask, None, bg_model, fg_model, iter_count, cv2.GC_INIT_WITH_MASK)
    else:
        raise ValueError("Invalid user_input: must be bounding box tuple or correctly formatted mask array.")

    # Extract the final binary mask
    refined = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
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
    Two-axis symmetry comparison (horizontal + vertical).
    Follows the standard ABCDE dermoscopy criterion by checking
    asymmetry on both axes and averaging the scores.
    """
    total = np.sum(mask > 0)
    if total == 0:
        return 0.0

    # 1. Find the exact boundaries of the lesion to ignore the surrounding canvas
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    
    # 2. Crop the mask tightly around the lesion
    cropped_mask = mask[y:y+h, x:x+w].astype(np.int32)
    
    # 3. Now the flips happen strictly across the lesion's own center axes
    diff_h = np.abs(cropped_mask - np.fliplr(cropped_mask))
    # We divide by (total * 2) because the maximum possible symmetric difference 
    # of a shape folded perfectly onto itself with zero overlap is 2x its area.
    asym_h = float(np.sum(diff_h > 0) / (total * 2.0))

    diff_v = np.abs(cropped_mask - np.flipud(cropped_mask))
    asym_v = float(np.sum(diff_v > 0) / (total * 2.0))

    return (asym_h + asym_v) / 2.0


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

    # Step 1 — refine segmentation
    refined_mask = refine_mask_grabcut(image, user_input)

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
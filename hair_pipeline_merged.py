"""
hair_pipeline_merged.py
=======================
Two-Stage Hair Removal Pipeline for ISIC Dataset
=================================================

STAGE 1 — HAIR DETECTION FILTER  (hair_detection.py logic)
  Multi-Feature Hybrid Approach:
    · Black-hat morphology     — isolates thin dark structures
    · Dark line detection      — Canny + Hough line density
    · FFT directional analysis — high-frequency anisotropic patterns
    · Thin structure connectivity — long connected dark paths
  Threshold: 0.30  (images with confidence >= 0.30 proceed to Stage 2)

STAGE 2 — DULLRAZOR HAIR REMOVAL  (dullrazor_full.py logic)
  Runs ONLY on images flagged as having hair in Stage 1:
    · Morphological closing minus original → hair signal
    · Adaptive threshold → binary hair mask
    · Telea inpainting at full resolution → reconstructed skin

OUTPUTS:
  · Processed (hair-removed) images in output directory
  · hair_pipeline_report.csv  — per-image results from both stages
  · images_with_hair.txt      — filenames flagged in Stage 1
  · images_without_hair.txt   — filenames cleared in Stage 1

USAGE:
  python hair_pipeline_merged.py
  python hair_pipeline_merged.py --input data/isic2019 --output data/isic2019_hairless
  python hair_pipeline_merged.py --test --n 20
  python hair_pipeline_merged.py --workers 8 --preview

OPTIONS:
  --input           Input dataset root dir         (default: data/isic2019)
  --output          Output directory               (default: data/isic2019_hairless)
  --detection-threshold  Stage 1 confidence threshold 0.0-1.0  (default: 0.30)
  --removal-threshold    Stage 2 hair %% to trigger inpainting  (default: 2.0)
  --workers         Parallel workers               (default: 4)
  --test            Run on a small sample only
  --n               Number of images for test mode (default: 10)
  --preview         Save a preview grid PNG for first 12 hairy images
  --copy-clean      Also copy hair-free images to output dir
  --resume          Skip images already in output dir
  --ext             Output extension: jpg or png   (default: jpg)
  --quality         JPEG save quality 1-100        (default: 95)
  --no-visuals      Skip saving Stage 1 visual result images
"""

import os
import sys
import csv
import argparse
import time
import warnings
import multiprocessing
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from lesion_processing import extract_features, OtsuSegmentation


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Stage 1 — Hair Detection Filter threshold
DETECTION_THRESHOLD = 0.30   # confidence >= this → image has hair → go to Stage 2

# Stage 2 — DullRazor removal threshold (hair pixel %)
REMOVAL_THRESHOLD = 2.0      # hair_pct >= this → apply inpainting


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessResult:
    name: str
    src_path: str
    dst_path: str = ""
    # Stage 1 fields
    detection_label: str = "NO_HAIR"
    detection_confidence: float = 0.0
    score_blackhat: float = 0.0
    score_lines: float = 0.0
    score_fft: float = 0.0
    score_thin: float = 0.0
    # Stage 2 fields
    hair_pct: float = 0.0
    removal_applied: bool = False
    # General
    status: str = "ok"        # ok | ok_no_hair | skipped | error
    error_msg: str = ""
    elapsed_ms: float = 0.0
    #lesion features
    lesion_area: float = 0.0
    lesion_circularity: float = 0.0
    lesion_asymmetry: float = 0.0
    lesion_color_variance: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — HAIR DETECTION FILTER
# Source: hair_detection.py
# ─────────────────────────────────────────────────────────────────────────────
# This stage determines WHETHER an image contains hair.
# It uses four independent feature extractors combined into a weighted ensemble.
# No hair removal is performed here.
# ══════════════════════════════════════════════════════════════════════════════

# ── Feature 1: Black-Hat Morphology ──────────────────────────────────────────
# Isolates thin dark structures against a lighter background.

def _blackhat_score(gray: np.ndarray) -> float:
    """
    Black-hat transform highlights dark thin structures (hair).
    Returns fraction of pixels likely to be hair.
    Uses three kernel sizes to catch both thin and thick strands.
    """
    scores = []
    for ksize in [17, 25, 35]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
        score = np.count_nonzero(mask) / mask.size
        scores.append(score)
    return max(scores)


# ── Feature 2: Dark Line Detection (Canny + Hough) ───────────────────────────
# Hair appears as thin elongated dark lines → detected via edge + line density.

def _dark_line_score(gray: np.ndarray) -> float:
    """
    Detects thin dark curvilinear lines via Canny edge detection and
    Probabilistic Hough Transform. Returns a normalised line-density score.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=8,
    )

    if lines is None:
        return 0.0

    h, w = gray.shape
    image_diagonal = np.sqrt(h ** 2 + w ** 2)
    total_length = sum(
        np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        for x1, y1, x2, y2 in lines[:, 0]
    )
    return min(total_length / (image_diagonal * 10), 1.0)


# ── Feature 3: FFT Directional Frequency Analysis ────────────────────────────
# Hair creates high-frequency anisotropic patterns detectable in the frequency domain.

def _fft_hair_score(gray: np.ndarray) -> float:
    """
    Analyses the frequency domain for elongated high-frequency patterns.
    Returns a score based on mid-to-high frequency ring energy concentration.
    """
    resized = cv2.resize(gray, (256, 256))
    f = np.fft.fft2(resized.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    cy, cx = 128, 128
    y_idx, x_idx = np.ogrid[:256, :256]
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    ring_mask = (dist > 20) & (dist < 100)

    ring_energy = magnitude[ring_mask].mean()
    total_energy = magnitude.mean()

    if total_energy == 0:
        return 0.0

    ratio = ring_energy / total_energy
    score = max(0.0, (ratio - 1.15) / 0.5)
    return min(score, 1.0)


# ── Feature 4: Thin Structure Connectivity ───────────────────────────────────
# Hair forms long connected dark paths detectable via connected components.

def _thin_structure_score(gray: np.ndarray) -> float:
    """
    Detects thin dark connected components typical of hair strands.
    Uses adaptive thresholding and morphological filtering to count strand-like
    connected components in both vertical and horizontal orientations.
    """
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=8,
    )

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    cleaned_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    cleaned_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    combined = cv2.bitwise_or(cleaned_v, cleaned_h)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    h, w = gray.shape
    min_area = (h * w) * 0.0003
    max_area = (h * w) * 0.05

    strand_count = sum(
        1 for i in range(1, num_labels)
        if min_area < stats[i, cv2.CC_STAT_AREA] < max_area
    )

    return min(strand_count / 30.0, 1.0)


# ── Master Detection Function ─────────────────────────────────────────────────

def stage1_detect_hair(image_path: str, threshold: float = DETECTION_THRESHOLD) -> dict:
    """
    STAGE 1 — Runs all four detection features and returns a combined result.

    This function ONLY determines whether hair is present.
    It does NOT remove hair. That is handled exclusively by Stage 2.

    Returns:
        dict with keys:
            has_hair       : bool
            confidence     : float (0–1)
            label          : 'HAIR' or 'NO_HAIR'
            scores         : dict of individual feature scores
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            "error": "Could not read image",
            "has_hair": False,
            "confidence": 0.0,
            "label": "NO_HAIR",
            "scores": {"blackhat": 0.0, "lines": 0.0, "fft": 0.0, "thin": 0.0},
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    s_blackhat = _blackhat_score(gray)
    s_lines    = _dark_line_score(gray)
    s_fft      = _fft_hair_score(gray)
    s_thin     = _thin_structure_score(gray)

    # Weighted ensemble — blackhat and thin structure are most reliable
    weights = {"blackhat": 0.35, "lines": 0.20, "fft": 0.15, "thin": 0.30}
    confidence = (
        weights["blackhat"] * s_blackhat
        + weights["lines"]  * s_lines
        + weights["fft"]    * s_fft
        + weights["thin"]   * s_thin
    )

    has_hair = confidence >= threshold

    return {
        "has_hair":   has_hair,
        "confidence": round(confidence, 4),
        "label":      "HAIR" if has_hair else "NO_HAIR",
        "scores": {
            "blackhat": round(s_blackhat, 4),
            "lines":    round(s_lines,    4),
            "fft":      round(s_fft,      4),
            "thin":     round(s_thin,     4),
        },
    }


# ── Stage 1 Visual Output ─────────────────────────────────────────────────────

def stage1_save_visual(image_path: str, result: dict, out_dir: str):
    """Save annotated detection result image for visual inspection (Stage 1)."""
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]
    max_dim = 400
    scale = max_dim / max(h, w)
    display = cv2.resize(img, (int(w * scale), int(h * scale)))
    dh, dw = display.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (dw, dh))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    blackhat = cv2.morphologyEx(gray_small, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    hair_overlay = display.copy()
    hair_overlay[hair_mask > 0] = [0, 0, 255]  # red = detected hair pixels
    blended = cv2.addWeighted(display, 0.6, hair_overlay, 0.4, 0)

    panel_w = 320
    panel = np.zeros((dh, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    label_color = (0, 200, 0) if result["has_hair"] else (100, 100, 255)
    label_text  = "HAIR DETECTED" if result["has_hair"] else "NO HAIR"
    cv2.putText(panel, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, label_color, 2)
    cv2.putText(panel, f"Confidence: {result['confidence']:.3f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    y = 115
    for name, val in result["scores"].items():
        bar_len = int(val * 240)
        cv2.rectangle(panel, (10, y), (10 + bar_len, y + 14), (80, 150, 80), -1)
        cv2.putText(panel, f"{name}: {val:.3f}", (10, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        y += 28

    cv2.putText(panel, Path(image_path).name[:32], (10, dh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    combined = np.hstack([blended, panel])
    fname = Path(image_path).stem + "_detection.jpg"
    subfolder = "hair" if result["has_hair"] else "no_hair"
    save_path = os.path.join(out_dir, "stage1_visuals", subfolder, fname)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, combined)


# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — DULLRAZOR HAIR REMOVAL
# Source: dullrazor_full.py
# ─────────────────────────────────────────────────────────────────────────────
# This stage runs ONLY on images that Stage 1 flagged as containing hair.
# It builds a precise pixel-level hair mask and inpaints the original image.
# Stage 1 detection logic is NOT used or repeated here.
# ══════════════════════════════════════════════════════════════════════════════

def stage2_build_hair_mask(
    image_bgr: np.ndarray,
    removal_threshold_pct: float = REMOVAL_THRESHOLD,
) -> tuple:
    """
    STAGE 2 — Builds a full-resolution binary hair mask via morphological closing.

    Method:
        Grayscale morphological closing fills thin dark hair strands.
        The difference (closed − original) isolates the hair signal.
        Adaptive thresholding converts the signal to a binary mask.

    Detection always runs on a 224×224 working copy for speed and
    resolution-independence. The mask is then scaled back to the
    original image dimensions using INTER_NEAREST to stay strictly binary.

    Returns:
        should_remove : bool  — True if hair_pct >= removal_threshold_pct
        hair_pct      : float — percentage of pixels classified as hair
        diff          : grayscale difference image at 224×224 (debug use)
        mask_full     : binary hair mask at original image resolution
    """
    h, w = image_bgr.shape[:2]

    # Working copy at 224×224 for morphological operations
    small = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Morphological closing fills thin dark strands
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Hair signal = difference between the closed image and the original
    diff = cv2.subtract(closed, gray)

    # Adaptive threshold — robust to uneven lighting and skin tone variation
    mask_small = cv2.adaptiveThreshold(
        diff,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=-2,
    )

    # Cleanup: close gaps and dilate to cover strand edges
    k3         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, k3)
    mask_small = cv2.dilate(mask_small, k3, iterations=1)

    # Scale mask back to original resolution — INTER_NEAREST keeps it binary
    if (h, w) == (224, 224):
        mask_full = mask_small
    else:
        mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    hair_pct = round(np.sum(mask_full > 0) / mask_full.size * 100, 4)
    should_remove = hair_pct >= removal_threshold_pct

    return should_remove, hair_pct, diff, mask_full


def stage2_inpaint(image_bgr: np.ndarray, mask_full: np.ndarray) -> np.ndarray:
    """
    STAGE 2 — Applies Telea inpainting to reconstruct skin texture under hair.

    Inpainting runs on the ORIGINAL full-resolution image.
    inpaintRadius is scaled proportionally from a base of 6 (tuned for 224px)
    so hair strands are correctly filled at any resolution.

    Returns the hairless image at the same size and quality as the input.
    """
    h, w   = image_bgr.shape[:2]
    scale  = max(h, w) / 224
    radius = max(3, int(round(6 * scale)))
    hairless = cv2.inpaint(image_bgr, mask_full, inpaintRadius=radius,
                           flags=cv2.INPAINT_TELEA)
    return hairless


# ══════════════════════════════════════════════════════════════════════════════
# WORKER — runs both stages for a single image (in subprocess)
# ══════════════════════════════════════════════════════════════════════════════

def process_single_image(
    src_path: str,
    dst_path: str,
    detection_threshold: float,
    removal_threshold_pct: float,
    copy_clean: bool,
    resume: bool,
    quality: int,
    ext: str,
    save_visuals: bool,
    vis_dir: str,
) -> ProcessResult:
    """
    Full two-stage worker for one image:
        Stage 1 → detect hair (multi-feature ensemble, threshold=detection_threshold)
        Stage 2 → if hair detected, build mask and inpaint (runs only when Stage 1 says HAIR)
    """
    t0 = time.perf_counter()
    src = Path(src_path)
    dst = Path(dst_path)

    result = ProcessResult(name=src.stem, src_path=src_path, dst_path=dst_path)

    result.hair_pct = 0.0
    result.removal_applied = False

    # Resume mode
    if resume and dst.exists():
        result.status = "skipped"
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    # ── STAGE 1: Hair Detection Filter ───────────────────────────────────────
    detection = stage1_detect_hair(src_path, threshold=detection_threshold)

    if "error" in detection:
        result.status = "error"
        result.error_msg = f"Stage1 read error: {detection['error']}"
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    result.detection_label      = detection["label"]
    result.detection_confidence = detection["confidence"]
    result.score_blackhat       = detection["scores"]["blackhat"]
    result.score_lines          = detection["scores"]["lines"]
    result.score_fft            = detection["scores"]["fft"]
    result.score_thin           = detection["scores"]["thin"]

    if save_visuals:
        try:
            stage1_save_visual(src_path, detection, vis_dir)
        except Exception:
            pass  # visual saving failure should never abort processing

    # Load image
    raw = cv2.imread(src_path)
    if raw is None:
        result.status = "error"
        result.error_msg = "Stage2: cv2.imread returned None"
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result
    
    # Default become no hair
    hairless = raw 
    result.status  = "ok_no_hair"

    # If no hair and copy_clean is enabled ,save directly
    if not detection["has_hair"]:
        if copy_clean:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                _save_image(raw, dst, quality, ext)
                result.dst_path = str(dst)
            except Exception as e:
                result.status = "error"
                result.error_msg = f"Copy clean failed: {e}"

    # ── STAGE 2: DullRazor Hair Removal ───────────────────────────────────
    if  detection["has_hair"]:
        try:
            should_remove, hair_pct, _diff, hair_mask = stage2_build_hair_mask(
                raw, removal_threshold_pct
            )
            result.hair_pct = hair_pct

        except Exception as e:
            result.status = "error"
            result.error_msg = f"Stage2 mask: {e}"
            result.elapsed_ms = (time.perf_counter() - t0) * 1000
            return result

        if  should_remove:
            try:
                hairless = stage2_inpaint(raw, hair_mask)
                result.removal_applied = True
                result.status = "ok"
            except Exception as e:
                result.status = "error"
                result.error_msg = f"Stage2 inpaint: {e}"
                result.elapsed_ms = (time.perf_counter() - t0) * 1000

    #Added the lesion processing step here to ensure it runs after hair removal and before saving the final image.
    try:
        h, w = hairless.shape[:2]
        
        # Instantiate our specific strategy
        segmenter = OtsuSegmentation()
        bbox = segmenter.segment(hairless)
        
        try:
            if bbox is not None: 
                # Pass the raw bounding box directly to the updated feature extractor
                lesion_features = extract_features(hairless, bbox)
            else:
                raise ValueError("OtsuSegmentation failed to find a valid lesion contour.")
            
        except Exception as e:
            result.error_msg += f" | lesion: {e}"
            h, w = hairless.shape[:2]
            fallback_bbox = (int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4))

            try:
                lesion_features = extract_features(hairless, fallback_bbox)

            except Exception :
                lesion_features = {
                    "area": 0.0,
                    "circularity": 0.0,
                    "asymmetry": 0.0,
                    "color_variance": 0.0
                }

        result.lesion_area = lesion_features["area"] / (h * w)
        result.lesion_circularity = lesion_features["circularity"]
        result.lesion_asymmetry = lesion_features["asymmetry"]
        result.lesion_color_variance = min(lesion_features["color_variance"] / (255.0 ** 2), 1.0)

    except Exception as e:
        #Log but DO NOT break pipeline
        result.error_msg += f" | lesion: {e}"

    # Save final hairless image (or original if no hair detected and copy_clean is False)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        _save_image(hairless, dst, quality, ext)
        result.dst_path = str(dst)
    except Exception as e:
        result.status = "error"
        result.error_msg = f"Stage2 save: {e}"
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    return result


def _save_image(img: np.ndarray, dst: Path, quality: int, ext: str):
    """Save image with appropriate compression settings."""
    if ext == "png":
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


# ══════════════════════════════════════════════════════════════════════════════
# PREVIEW GRID (Stage 2 — shows before/after for hairy images)
# ══════════════════════════════════════════════════════════════════════════════

def save_preview_grid(
    hairy_paths: list,
    removal_threshold_pct: float,
    save_path: str = "pipeline_preview.png",
):
    """Generate a dark-themed 4-column preview grid for the first 12 hairy images."""
    import matplotlib.pyplot as plt

    paths = hairy_paths[:12]
    N = len(paths)
    if N == 0:
        print("  No hairy images to preview.")
        return

    COL_TITLES = [
        "Original",
        "Hair Signal\n(closing − original)",
        "Hair Mask\n(adaptive threshold)",
        "Hairless Result",
    ]
    fig, axes = plt.subplots(N, 4, figsize=(20, N * 4))
    if N == 1:
        axes = axes[np.newaxis, :]
    fig.patch.set_facecolor("#0e0e0e")

    for col, title in enumerate(COL_TITLES):
        axes[0, col].set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)

    for row, img_path in enumerate(paths):
        raw = cv2.imread(str(img_path))
        if raw is None:
            continue
        _, hair_pct, diff, mask_full = stage2_build_hair_mask(raw, removal_threshold_pct)
        hairless = stage2_inpaint(raw, mask_full)

        orig_disp     = cv2.resize(raw,      (224, 224), interpolation=cv2.INTER_AREA)
        hairless_disp = cv2.resize(hairless,  (224, 224), interpolation=cv2.INTER_AREA)
        mask_disp     = cv2.resize(mask_full, (224, 224), interpolation=cv2.INTER_NEAREST)

        panels = [
            (cv2.cvtColor(orig_disp,     cv2.COLOR_BGR2RGB), None),
            (diff,                                            "hot"),
            (mask_disp,                                       "gray"),
            (cv2.cvtColor(hairless_disp, cv2.COLOR_BGR2RGB), None),
        ]

        for col, (panel_img, cmap) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel_img, cmap=cmap) if cmap else ax.imshow(panel_img)
            ax.set_facecolor("#0e0e0e")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a2a")

        axes[row, 0].set_ylabel(
            f"{img_path.stem}\nHair: {hair_pct}%",
            color="white", fontsize=8, labelpad=6,
        )

    fig.suptitle(
        f"Hair Removal Pipeline — Preview  (detection≥{DETECTION_THRESHOLD}, removal≥{removal_threshold_pct}%)\n"
        "Stage1: Multi-Feature Detection → Stage2: Closing → Adaptive Threshold → Telea Inpainting",
        color="white", fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Preview saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def scan_dataset(root: Path) -> list:
    """Recursively find all supported image files in the dataset directory."""
    paths = []
    for ext in SUPPORTED_EXT:
        paths.extend(root.rglob(f"*{ext}"))
        paths.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def build_dst_path(src: Path, input_root: Path, output_root: Path, ext: str) -> Path:
    """Mirror source folder structure under output_root with a new extension."""
    rel = src.relative_to(input_root)
    return output_root / rel.with_suffix(f".{ext}")


class ProgressBar:
    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        self.done  = 0
        self.t0    = time.perf_counter()

    def update(self, n: int = 1):
        self.done += n
        pct    = self.done / self.total
        filled = int(self.width * pct)
        bar    = "█" * filled + "░" * (self.width - filled)
        elapsed = time.perf_counter() - self.t0
        eta = (elapsed / self.done) * (self.total - self.done) if self.done > 0 else 0
        sys.stdout.write(f"\r  [{bar}] {self.done}/{self.total}  ETA {eta:.0f}s  ")
        sys.stdout.flush()

    def finish(self):
        elapsed = time.perf_counter() - self.t0
        sys.stdout.write(
            f"\r  {'█' * self.width}  {self.total}/{self.total}  Done in {elapsed:.1f}s\n"
        )
        sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage Hair Detection + Removal Pipeline for ISIC Dataset"
    )
    parser.add_argument("--input",      default="data/isic2019",
                        help="Dataset root directory")
    parser.add_argument("--output",     default="data/isic2019_hairless",
                        help="Output directory")
    parser.add_argument("--detection-threshold", type=float, default=DETECTION_THRESHOLD,
                        help=f"Stage 1 confidence threshold 0.0-1.0 (default: {DETECTION_THRESHOLD})")
    parser.add_argument("--removal-threshold",   type=float, default=REMOVAL_THRESHOLD,
                        help=f"Stage 2 hair %% to trigger inpainting (default: {REMOVAL_THRESHOLD})")
    parser.add_argument("--workers",    type=int,   default=4,
                        help="Parallel worker processes (default: 4)")
    parser.add_argument("--test",       action="store_true",
                        help="Run on a small sample only")
    parser.add_argument("--n",          type=int,   default=10,
                        help="Number of images for test mode (default: 10)")
    parser.add_argument("--preview",    action="store_true",
                        help="Save preview grid PNG for first 12 hairy images")
    parser.add_argument("--copy-clean", action="store_true",
                        help="Also copy hair-free images to output dir")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip images already processed in output dir")
    parser.add_argument("--ext",        default="jpg", choices=["jpg", "png"],
                        help="Output image extension (default: jpg)")
    parser.add_argument("--quality",    type=int,   default=95,
                        help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--no-visuals", action="store_true",
                        help="Skip saving Stage 1 visual detection images")
    args = parser.parse_args()

    input_root  = Path(args.input)
    output_root = Path(args.output)

    print()
    print("=" * 65)
    print("  Two-Stage Hair Detection + Removal Pipeline")
    print("=" * 65)
    print(f"  Input                 : {input_root.resolve()}")
    print(f"  Output                : {output_root.resolve()}")
    print(f"  Stage 1 threshold     : {args.detection_threshold}  (confidence, multi-feature)")
    print(f"  Stage 2 threshold     : {args.removal_threshold}%  (hair pixel coverage)")
    print(f"  Workers               : {args.workers}")
    print(f"  Resume                : {args.resume}")
    print(f"  Copy clean images     : {args.copy_clean}")
    print(f"  Output ext            : .{args.ext}  (quality={args.quality})")
    print(f"  Stage 1 visuals       : {not args.no_visuals}")
    print()

    if not input_root.exists():
        print(f"  ERROR: Input directory not found: {input_root.resolve()}")
        sys.exit(1)

    # ── Scan and optionally limit to test subset ──────────────────────────────
    print("  Scanning dataset...")
    all_images = scan_dataset(input_root)
    if not all_images:
        print(f"  ERROR: No images found in {input_root.resolve()}")
        sys.exit(1)

    if args.test:
        images = all_images[: args.n]
        print(f"  Found {len(all_images)} images — TEST MODE: processing first {len(images)}.\n")
    else:
        images = all_images
        print(f"  Found {len(images)} images.\n")

    # ── Build job list ────────────────────────────────────────────────────────
    vis_dir = str(output_root)
    jobs = [(str(src), str(build_dst_path(src, input_root, output_root, args.ext)))
            for src in images]

    # ── Process all images ────────────────────────────────────────────────────
    report_rows  = []
    hairy_paths  = []
    names_hair   = []
    names_no_hair = []
    counters     = {"ok": 0, "ok_no_hair": 0, "skipped": 0, "error": 0}
    total_hair_pct = []
    t_start      = time.perf_counter()

    print(f"  Processing with {args.workers} worker(s)...\n")
    pbar = ProgressBar(len(jobs))

    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(
                process_single_image,
                src, dst,
                args.detection_threshold,
                args.removal_threshold,
                args.copy_clean,
                args.resume,
                args.quality,
                args.ext,
                not args.no_visuals,
                vis_dir,
            ): (src, dst)
            for src, dst in jobs
        }

        for future in as_completed(futures):
            try:
                res: ProcessResult = future.result()
            except Exception as e:
                src, dst = futures[future]
                res = ProcessResult(
                    name=Path(src).stem, src_path=src, dst_path=dst,
                    status="error", error_msg=str(e),
                )

            report_rows.append(res)
            status_key = res.status if res.status in counters else "error"
            counters[status_key] += 1

            if res.detection_label == "HAIR":
                hairy_paths.append(Path(res.src_path))
                names_hair.append(Path(res.src_path).name)
                if res.hair_pct > 0:
                    total_hair_pct.append(res.hair_pct)
            else:
                names_no_hair.append(Path(res.src_path).name)

            pbar.update()

    pbar.finish()

    # ── Save CSV report ───────────────────────────────────────────────────────
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "hair_pipeline_report.csv"

    report_rows.sort(key=lambda r: r.name)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name", "src_path", "dst_path",
            "stage1_label", "stage1_confidence",
            "score_blackhat", "score_lines", "score_fft", "score_thin",
            "stage2_hair_pct", "stage2_removal_applied",
            "status", "error_msg", "elapsed_ms","lesion_area",
            "lesion_circularity","lesion_asymmetry","lesion_color_variance"
        ])
        for r in report_rows:
            writer.writerow([
                r.name, r.src_path, r.dst_path,
                r.detection_label, r.detection_confidence,
                r.score_blackhat, r.score_lines, r.score_fft, r.score_thin,
                r.hair_pct, r.removal_applied,
                r.status, r.error_msg, f"{r.elapsed_ms:.1f}",
                r.lesion_area,r.lesion_circularity,r.lesion_asymmetry,
                r.lesion_color_variance
            ])

    # ── Save filtered image lists ─────────────────────────────────────────────
    with open(output_root / "images_with_hair.txt", "w") as f:
        f.write("\n".join(names_hair))
    with open(output_root / "images_without_hair.txt", "w") as f:
        f.write("\n".join(names_no_hair))

    # ── Summary ───────────────────────────────────────────────────────────────
    total_sec = time.perf_counter() - t_start
    n_hairy   = len(hairy_paths)
    n_total   = len(jobs)
    avg_hair  = round(np.mean(total_hair_pct), 2) if total_hair_pct else 0.0

    print()
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Total images          : {n_total}")
    print(f"  Stage 1 — HAIR        : {n_hairy}  ({100 * n_hairy / max(n_total, 1):.1f}%)")
    print(f"  Stage 1 — NO HAIR     : {n_total - n_hairy}")
    print(f"  Stage 2 — Removed     : {counters['ok']}")
    print(f"  Avg hair coverage     : {avg_hair}%  (on hairy images)")
    print(f"  Skipped (resume)      : {counters['skipped']}")
    print(f"  Errors                : {counters['error']}")
    print(f"  Total time            : {total_sec:.1f}s  ({total_sec / max(n_total, 1) * 1000:.1f} ms/image)")
    print(f"\n  Output files:")
    print(f"    {report_path}")
    print(f"    {output_root / 'images_with_hair.txt'}")
    print(f"    {output_root / 'images_without_hair.txt'}")
    if not args.no_visuals:
        print(f"    {output_root / 'stage1_visuals' / 'hair'}/*.jpg")
        print(f"    {output_root / 'stage1_visuals' / 'no_hair'}/*.jpg")

    # ── Preview grid ──────────────────────────────────────────────────────────
    if args.preview:
        print("\n  Generating Stage 2 preview grid...")
        preview_path = str(output_root / "pipeline_preview.png")
        save_preview_grid(hairy_paths[:12], args.removal_threshold, save_path=preview_path)

    if counters["error"] > 0:
        print(f"\n  ⚠  {counters['error']} errors — check {report_path} for details.")

    print("\n  Done! ✓")
    print(f"  Hairless images → {output_root.resolve()}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

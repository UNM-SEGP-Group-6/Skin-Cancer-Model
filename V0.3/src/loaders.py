
from pathlib import Path
import pandas as pd
import numpy as np

import src.config as src_config

def _scan_images(folders, extensions=(".jpg", ".png")):
    """
    Scan multiple folders and return a mapping of image identifiers to file paths.

    Parameters:
        folders (list): List of directory paths to scan.
        extensions (tuple): Allowed file extensions (case-insensitive).

    Returns:
        dict: Mapping {image_id: full_path} for each found image.
              Each image is stored under both its stem and full filename as keys.
    """
    paths = {}
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists(): continue
        for f in folder_path.iterdir():
            if f.suffix.lower() in extensions:
                name = f.stem
                paths[name] = f
                paths[f.name] = f
    return paths

def load_ham10000():
    """
    Load the HAM10000 dataset (10015 dermoscopic images, 7 classes).

    Reads metadata CSV, scans image directories, maps native labels
    to the unified class scheme, and returns a standardized DataFrame.

    Returns:
        Tuple[pd.DataFrame, dict]: (metadata DataFrame, {image_id: path} mapping).
    """

    meta = pd.read_csv(src_config.PATHS["ham10000"]["metadata"])

    # Scan both image folders for HAM10000
    image_paths = _scan_images([src_config.PATHS["ham10000"]["images_1"], src_config.PATHS["ham10000"]["images_2"]])
    df = meta.copy()
    df["label"] = df["dx"].map(src_config.LABEL_MAPPINGS["ham10000"]) # Ensure we map to canonical labels
    df["dataset"] = "ham10000"

    # Standardize localization strings to unified vocabulary
    loc_map = src_config.LOCALIZATION_MAPPINGS["ham10000"]
    df["localization"] = df["localization"].str.lower().str.strip().map(
        lambda x: loc_map.get(x, "unknown") if pd.notna(x) else "unknown"
    )

    # Keep only samples with available images
    df = df[df["image_id"].isin(image_paths)].copy()
    return df, image_paths

def load_isic2019():
    """
    Load the ISIC 2019 dataset (25,000+ dermoscopic images, 8 classes).

    Merges metadata with ground truth labels, maps class names to
    the unified scheme, and handles column name differences.

    Returns:
        Tuple[pd.DataFrame, dict]: (metadata DataFrame, {image_id: path} mapping).
    """

    meta = pd.read_csv(src_config.PATHS["isic_2019"]["metadata"])
    labels = pd.read_csv(src_config.PATHS["isic_2019"]["labels"])
    
    # Merge metadata with one-hot ground truth labels
    df = meta.merge(labels, on="image")

    # Determine label from the one-hot columns
    class_cols = [c for c in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"] if c in df.columns]
    df["label"] = df[class_cols].idxmax(axis=1).map(src_config.LABEL_MAPPINGS["isic_2019"])
    df["image_id"] = df["image"]
    df["dataset"] = "isic_2019"

    # Standardize column names across datasets
    df["age"] = df.get("age_approx", pd.Series(dtype=float))
    if "sex" not in df.columns: df["sex"] = np.nan

    # Standardize localization strings to unified vocabulary
    loc_map = src_config.LOCALIZATION_MAPPINGS["isic_2019"]
    raw_loc = df.get("anatom_site_general", pd.Series(dtype=str))
    df["localization"] = raw_loc.str.lower().str.strip().map(
        lambda x: loc_map.get(x, "unknown") if pd.notna(x) else "unknown"
    )

    # Scan image directory
    image_paths = _scan_images([src_config.PATHS["isic_2019"]["images"]])

    # Strip _downsampled suffix if present in filenames
    extra = {k.replace("_downsampled", ""): v for k, v in image_paths.items() if "_downsampled" in k}
    image_paths.update(extra)

    # Filter to samples with valid images and labels
    df = df[df["image_id"].isin(image_paths) & df["label"].notna()].copy()
    return df, image_paths

def load_ph2():
    """
    Load the PH2 dataset (200 dermoscopic images, 2 effective classes: nv and mel).

    Parses the Excel metadata file and constructs image paths
    following PH2's nested directory structure.

    Returns:
        Tuple[pd.DataFrame, dict]: (metadata DataFrame, {image_id: path} mapping).
    """

    # PH2 metadata starts at row 12 in the Excel file
    ph2_excel = pd.read_excel(src_config.PATHS["ph2"]["metadata"], header=12)
    ph2_excel.columns = ["image_name", "histological_diagnosis", "common_nevus", "atypical_nevus", "melanoma",
                         "asymmetry", "pigment_network", "dots_globules", "streaks", "regression_areas",
                         "blue_whitish_veil", "white", "colors", "col13", "col14", "col15", "col16"]
    samples, image_paths = [], {}
    base = src_config.PATHS["ph2"]["images"]
    for _, row in ph2_excel.iterrows():
        img_id = row["image_name"]
        if pd.isna(img_id): continue
        
        # Determine label from one-hot-style columns
        if   row["melanoma"]       == "X": label = "mel"
        elif row["atypical_nevus"] == "X": label = "nv"
        elif row["common_nevus"]   == "X": label = "nv"
        else: continue

        # PH2 has a nested path structure: <img_id>/<img_id>_Dermoscopic_Image/<img_id>.bmp
        img_path = Path(base) / img_id / f"{img_id}_Dermoscopic_Image" / f"{img_id}.bmp"
        if img_path.exists():
            samples.append({"image_id": img_id, "label": label, "age": np.nan, "sex": np.nan, "localization": np.nan, "dataset": "ph2"})
            image_paths[img_id] = img_path
    return pd.DataFrame(samples), image_paths

def load_pad_ufes20():
    """
    Load the PAD-UFES-20 dataset (clinical smartphone images, 6 classes).

    This dataset uses clinical (non-dermoscopic) images, representing
    a significant domain shift from the training data.

    Returns:
        Tuple[pd.DataFrame, dict]: (metadata DataFrame, {image_id: path} mapping).
    """

    meta = pd.read_csv(src_config.PATHS["pad_ufes_20"]["metadata"])
    label_map = src_config.LABEL_MAPPINGS["pad_ufes_20"]
    
    # Scan all three image folders (supports .png and .jpg)
    all_imgs = _scan_images(
        [src_config.PATHS["pad_ufes_20"]["images_1"], src_config.PATHS["pad_ufes_20"]["images_2"], src_config.PATHS["pad_ufes_20"]["images_3"]],
        extensions=(".png", ".jpg")
    )

    # Auto-detect column names (vary across dataset versions)
    img_col  = next((c for c in ["img_id", "image_id", "image"] if c in meta.columns), None)
    diag_col = next((c for c in ["diagnostic", "diagnosis", "label", "dx"] if c in meta.columns), None)
    sex_col  = next((c for c in ["gender", "sex"] if c in meta.columns), None)
    loc_col  = next((c for c in ["region", "localization"] if c in meta.columns), None)

    samples, image_paths = [], {}
    for _, row in meta.iterrows():
        img_id = str(row[img_col])
        diag = row[diag_col]

        if pd.isna(diag): continue
        # Map native diagnosis to unified label
        label = label_map.get(str(diag).upper())

        if not label: continue
        # Try multiple filename formats to find the image
        img_path = all_imgs.get(img_id) or all_imgs.get(img_id + ".png") or all_imgs.get(img_id + ".jpg")
        if not img_path or not Path(img_path).exists(): continue
        
        # Normalize sex values to standard format
        sex = np.nan
        if sex_col and pd.notna(row.get(sex_col)):
            sv = str(row[sex_col]).lower()
            sex = "male" if sv in ["male", "m"] else ("female" if sv in ["female", "f"] else np.nan)
        samples.append({
            "image_id": img_id, "label": label,
            "age": row.get("age", np.nan) if pd.notna(row.get("age", np.nan)) else np.nan,
            "sex": sex,
            "localization": row.get(loc_col, np.nan) if loc_col and pd.notna(row.get(loc_col)) else np.nan,
            "dataset": "pad_ufes_20"
        })
        image_paths[img_id] = img_path
    return pd.DataFrame(samples), image_paths

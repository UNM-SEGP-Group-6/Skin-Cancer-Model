
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


def _map_midas_location(loc_string):
    """
    Map a granular MRA-MIDAS location string to a unified localization category.

    MRA-MIDAS has 428 unique location strings (e.g., "l upper back", "r cheek",
    "left forearm"). This function uses keyword matching with a priority-ordered
    list to map them to the 11 unified localization categories.

    Keyword order matters: more specific terms are checked before general ones
    (e.g., "forearm" before "arm", "forehead" before "head").

    Parameters:
        loc_string (str): Raw MRA-MIDAS location value.

    Returns:
        str: Unified localization category (e.g., "head_neck", "upper_extremity").
    """
    if pd.isna(loc_string):
        return "unknown"

    loc = str(loc_string).lower().strip()
    if not loc:
        return "unknown"

    # --- Keyword matching (order matters: specific before general) ---

    # Palms & soles (check before "arm"/"leg" to catch "dorsal hand", "finger", etc.)
    if any(kw in loc for kw in ["finger", "thumb", "palm", "dorsal hand", "knuckle"]):
        return "palms_soles"
    if any(kw in loc for kw in ["toe", "sole", "heel", "dorsal foot", "plantar"]):
        return "palms_soles"
    if loc in ["hand", "r hand", "l hand", "right hand", "left hand"]:
        return "palms_soles"
    if loc in ["foot", "r foot", "l foot", "right foot", "left foot"]:
        return "palms_soles"

    # Head & neck (check "forehead" before "head", "forearm" hasn't matched yet)
    if any(kw in loc for kw in [
        "scalp", "temple", "forehead", "cheek", "ear", "nose", "lip",
        "jaw", "chin", "eyebrow", "eyelid", "face", "occiput", "orbit",
        "parietal", "periauricular", "preauricular", "postauricular",
        "nasal", "tragus", "antitragus", "helix", "ala",
        "mandible", "maxilla", "malar", "perioral", "glabella",
        "brow", "canthal", "canthus",
    ]):
        return "head_neck"
    if "neck" in loc:
        return "head_neck"
    if "head" in loc:
        return "head_neck"

    # Upper extremity (check "forearm" before "arm")
    if any(kw in loc for kw in ["forearm", "elbow", "wrist", "antecubital"]):
        return "upper_extremity"
    if "shoulder" in loc:
        return "upper_extremity"
    if "arm" in loc and "forearm" not in loc:
        return "upper_extremity"
    if "upper arm" in loc:
        return "upper_extremity"
    if "axilla" in loc or "axillary" in loc:
        return "upper_extremity"

    # Lower extremity (check before "ankle" which contains "an")
    if any(kw in loc for kw in [
        "shin", "thigh", "knee", "calf", "ankle", "lower leg",
        "popliteal", "pretibial", "malleolus", "tibial",
    ]):
        return "lower_extremity"
    if "leg" in loc:
        return "lower_extremity"

    # Oral/genital/buttock
    if any(kw in loc for kw in [
        "genital", "groin", "buttock", "gluteal", "perineal",
        "inguinal", "pubic", "perianal", "scrotal", "vulvar",
    ]):
        return "oral_genital"

    # Abdomen (check before "torso" catch-all)
    if any(kw in loc for kw in ["abdomen", "abdominal", "umbilicus", "umbilical", "periumbilical"]):
        return "abdomen"

    # Torso anterior
    if any(kw in loc for kw in [
        "chest", "anterior", "clavicle", "sternum", "sternal",
        "breast", "inframammary", "supraclavicular", "infraclavicular",
        "pectoral", "presternal",
    ]):
        return "torso_anterior"

    # Torso lateral
    if any(kw in loc for kw in ["flank", "lateral"]):
        return "torso_lateral"

    # Torso posterior (back)
    if "back" in loc:
        return "torso_posterior"

    # Torso general
    if any(kw in loc for kw in ["trunk", "torso"]):
        return "torso"

    # Fallback
    return "unknown"


def load_mra_midas():
    """
    Load the MRA-MIDAS dataset (3416 clinical smartphone images, 3 classes).

    Reads the Excel metadata file, maps pathology diagnoses to the 3-class
    scheme (benign/malignant/uncertain), maps granular body locations to the
    unified localization vocabulary, and scans the flat image directory.

    Rows with unmappable pathology labels (NaN, 0, or values not in
    LABEL_MAPPINGS) are skipped — this naturally excludes control images
    which have null pathology labels.

    Returns:
        Tuple[pd.DataFrame, dict]: (metadata DataFrame, {image_id: path} mapping).

    DataFrame columns: image_id, label, age, sex, localization, dataset
    """

    meta = pd.read_excel(src_config.PATHS["mra_midas"]["metadata"])
    label_map = src_config.LABEL_MAPPINGS["mra_midas"]

    # Scan image directory for .jpg files
    image_paths = _scan_images(
        [src_config.PATHS["mra_midas"]["images"]],
        extensions=(".jpg",)
    )

    samples = []
    missing_images = []
    for _, row in meta.iterrows():
        # Extract file name and build image_id (stem without extension)
        file_name = row.get("midas_file_name")
        if pd.isna(file_name):
            continue
        img_id = str(file_name).replace(".jpg", "")

        # Map pathology label to 3-class scheme
        midas_path = row.get("midas_path")
        if pd.isna(midas_path) or midas_path == 0:
            continue
        label = label_map.get(str(midas_path).strip().lower())
        if label is None:
            continue

        # Check image file exists
        img_path = image_paths.get(img_id) or image_paths.get(file_name)
        if not img_path or not Path(img_path).exists():
            missing_images.append(str(file_name))
            continue

        # Normalize sex
        sex = np.nan
        gender = row.get("midas_gender")
        if pd.notna(gender):
            sv = str(gender).lower().strip()
            sex = "male" if sv == "male" else ("female" if sv == "female" else np.nan)

        # Map granular location to unified scheme
        localization = _map_midas_location(row.get("midas_location"))

        # Normalize age
        age = row.get("midas_age", np.nan)
        if pd.notna(age):
            age = float(age)
        else:
            age = np.nan

        samples.append({
            "image_id": img_id,
            "label": label,
            "age": age,
            "sex": sex,
            "localization": localization,
            "dataset": "mra_midas",
        })
        image_paths[img_id] = img_path

    df = pd.DataFrame(samples)
    print(f"  MRA-MIDAS loaded: {len(df)} samples")
    if missing_images:
        print(f"    WARNING: Dropped {len(missing_images)} samples because their image files were missing from the directory.")
        try:
            with open("missing_images_mra_midas.txt", "w") as f:
                f.write("\n".join(missing_images))
            print("    Saved list of missing files to 'missing_images_mra_midas.txt'")
        except Exception as e:
            print(f"    Failed to save missing files list: {e}")
    if len(df) > 0:
        print(f"    Label distribution: {df['label'].value_counts().to_dict()}")
    return df, image_paths

def load_ph2():
    """
    Load the PH2 dataset (200 dermoscopic images, 2 effective classes: benign and malignant).

    Used as an external holdout test set for cross-domain evaluation
    (dermoscopic vs clinical smartphone images).

    Parses the Excel metadata file and constructs image paths
    following PH2's nested directory structure.

    Returns:
        Tuple[pd.DataFrame, dict]: (metadata DataFrame, {image_id: path} mapping).
    """

    label_map = src_config.LABEL_MAPPINGS["ph2"]

    # PH2 metadata starts at row 12 in the Excel file
    ph2_excel = pd.read_excel(src_config.PATHS["ph2"]["metadata"], header=12)
    ph2_excel.columns = ["image_name", "histological_diagnosis", "common_nevus", "atypical_nevus", "melanoma",
                         "asymmetry", "pigment_network", "dots_globules", "streaks", "regression_areas",
                         "blue_whitish_veil", "white", "colors", "col13", "col14", "col15", "col16"]
    samples, image_paths = [], {}
    missing_images = []
    base = src_config.PATHS["ph2"]["images"]
    for _, row in ph2_excel.iterrows():
        img_id = row["image_name"]
        if pd.isna(img_id): continue
        
        # Determine label from one-hot-style columns, then map to 3-class scheme
        if   row["melanoma"]       == "X": raw_label = "melanoma"
        elif row["atypical_nevus"] == "X": raw_label = "atypical_nevus"
        elif row["common_nevus"]   == "X": raw_label = "common_nevus"
        else: continue

        label = label_map.get(raw_label)
        if label is None: continue

        # PH2 has a nested path structure: <img_id>/<img_id>_Dermoscopic_Image/<img_id>.bmp
        img_path = Path(base) / img_id / f"{img_id}_Dermoscopic_Image" / f"{img_id}.bmp"
        if img_path.exists():
            samples.append({"image_id": img_id, "label": label, "age": np.nan, "sex": np.nan, "localization": np.nan, "dataset": "ph2"})
            image_paths[img_id] = img_path
        else:
            missing_images.append(str(img_id))

    df = pd.DataFrame(samples)
    print(f"  PH2 loaded: {len(df)} samples")
    if missing_images:
        print(f"    WARNING: Dropped {len(missing_images)} samples because their image files were missing from the directory.")
        try:
            with open("missing_images_ph2.txt", "w") as f:
                f.write("\n".join(missing_images))
            print("    Saved list of missing files to 'missing_images_ph2.txt'")
        except Exception as e:
            print(f"    Failed to save missing files list: {e}")
    if len(df) > 0:
        print(f"    Label distribution: {df['label'].value_counts().to_dict()}")
    return df, image_paths


def load_pad_ufes20():
    """
    Load the PAD-UFES-20 dataset (clinical smartphone images, 2 effective classes: benign and malignant).

    This dataset uses clinical (non-dermoscopic) images, same domain as MRA-MIDAS.
    Combined with MRA-MIDAS to form the training pool.

    Returns:
        Tuple[pd.DataFrame, dict]: (metadata DataFrame, {image_id: path} mapping).
    """

    meta = pd.read_csv(src_config.PATHS["pad_ufes_20"]["metadata"])
    label_map = src_config.LABEL_MAPPINGS["pad_ufes_20"]
    loc_map = src_config.LOCALIZATION_MAPPINGS["pad_ufes_20"]
    
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
    missing_images = []
    for _, row in meta.iterrows():
        img_id = str(row[img_col])
        diag = row[diag_col]

        if pd.isna(diag): continue
        # Map native diagnosis to 3-class label
        label = label_map.get(str(diag).upper())

        if not label: continue
        # Try multiple filename formats to find the image
        img_path = all_imgs.get(img_id) or all_imgs.get(img_id + ".png") or all_imgs.get(img_id + ".jpg")
        if not img_path or not Path(img_path).exists():
            missing_images.append(str(img_id))
            continue
        
        # Normalize sex values to standard format
        sex = np.nan
        if sex_col and pd.notna(row.get(sex_col)):
            sv = str(row[sex_col]).lower()
            sex = "male" if sv in ["male", "m"] else ("female" if sv in ["female", "f"] else np.nan)

        # Map localization to unified scheme
        raw_loc = row.get(loc_col, np.nan) if loc_col else np.nan
        if pd.notna(raw_loc):
            localization = loc_map.get(str(raw_loc).upper().strip(), "unknown")
        else:
            localization = "unknown"

        samples.append({
            "image_id": img_id, "label": label,
            "age": row.get("age", np.nan) if pd.notna(row.get("age", np.nan)) else np.nan,
            "sex": sex,
            "localization": localization,
            "dataset": "pad_ufes_20"
        })
        image_paths[img_id] = img_path

    df = pd.DataFrame(samples)
    print(f"  PAD-UFES-20 loaded: {len(df)} samples")
    if missing_images:
        print(f"    WARNING: Dropped {len(missing_images)} samples because their image files were missing from the directory.")
        try:
            with open("missing_images_pad_ufes_20.txt", "w") as f:
                f.write("\n".join(missing_images))
            print("    Saved list of missing files to 'missing_images_pad_ufes_20.txt'")
        except Exception as e:
            print(f"    Failed to save missing files list: {e}")
    if len(df) > 0:
        print(f"    Label distribution: {df['label'].value_counts().to_dict()}")
    return df, image_paths

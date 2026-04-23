import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Tuple

@dataclass
class DataConfig:
    """Data loading and preprocessing parameters."""
    img_size: int = 484             # Fixed Input Size
    batch_size: int = 8             # Batch Size of images
    num_workers: int = 0            # Set to 0 on windows, 4+ on Linux/Colab/Kaggle
    test_size: float = 0.30         # 70/15/15 splits
    val_split: float = 0.50         # split of temp -> val/test (30=15/15)

@dataclass
class AugmentConfig:
    """Training augmentation parameters."""
    rotation: int = 30                                  # Rotation range (degrees, positive & negative)
    brightness: float = 0.3                             # Brightness range (fractional)
    contrast: float = 0.25                               # Contrast range (fractional)
    saturation: float = 0.25                             # Saturation range (fractional)
    hue: float = 0.15                                    # Hue range (fractional)
    translate: Tuple[float, float] = (0.1, 0.1)         # Translation range (fractional)
    scale: Tuple[float, float] = (0.9, 1.1)             # Scale range (fractional)
    flip_p: float = 0.5                                 # Probability of horizontal & vertical flip
                           

@dataclass
class ModelConfig:
    """Model architecture parameters."""
    backbone: str = "efficientnet_b1"                   # Backbone Model to use
    pretrained: bool = True                             # Pretrained weights
    freeze_backbone: bool = True                        # Freeze backbone layers during early training (epochs 1 - unfreeze_epoch)   

    meta_hidden: Tuple[int, int] = (64, 32)             # Hidden layer sizes for metadata processing
    meta_dropout: float = 0.3                           # Random Dropout Percentage for metadata processing

    classifier_hidden: int = 512                        # Hidden layer size for classifier
    classifier_dropout: float = 0.4                     # Random Dropout Percentage for classifier         

@dataclass
class TrainConfig:
    """Training loop parameters."""
    epochs: int = 35                            # Trains for "Epoch" passes through the dataset.
    lr: float = 1e-3                            # Learning Rate (Starts at 0.001)
    weight_decay: float = 1e-4                  # Weight Decay (Regularization)
    accumulation_steps: int = 4                 # Gradient accumulation steps. Effective batch = batch_size * accumulation_steps.

    unfreeze_epoch: int = 5                     # Epoch markpoint to unfreeze backbone model.
    backbone_lr_factor: float = 0.1             # Learning rate factor for unfrozen backbone.

    patience: int = 10                          # Number of epochs to wait before early stopping.
    min_delta: float = 0.001                    # Minimum change in the monitored quantity to qualify as an improvement.
    label_smoothing: float = 0.15               # Label smoothing factor for FocalLoss (0.0 = disabled).

@dataclass
class Config:
    """Master configuration combining all parameter groups."""
    data:   DataConfig = field(default_factory = DataConfig)
    aug: AugmentConfig = field(default_factory = AugmentConfig)
    model: ModelConfig = field(default_factory = ModelConfig)
    train: TrainConfig = field(default_factory = TrainConfig)

cfg = Config()

DATA_ROOT = Path(".") # Default to current directory

PATHS = {}

def set_num_workers() -> int:
    if os.name == 'nt':
        cfg.data.num_workers = 0
        return 0
    else: 
        cfg.data.num_workers = 4
        return 4

def detect_os_set_paths():
    global PATHS
    
    # Windows Pathing
    if os.name == 'nt':
        print("Running on Windows & Using Windows Pathing, Update it as need be.\n")

        PATHS = {
            "mra_midas": {
                "base"     : DATA_ROOT / "midasmultimodalimagedatasetforaibasedskincancer",
                "metadata" : DATA_ROOT / "midasmultimodalimagedatasetforaibasedskincancer" / "release_midas.xlsx",
                "images"   : DATA_ROOT / "midasmultimodalimagedatasetforaibasedskincancer",
            },
            "pad_ufes_20": {
                "base"     : DATA_ROOT / "PAD-UFES-20",
                "metadata" : DATA_ROOT / "PAD-UFES-20" / "metadata.csv",
                "images_1" : DATA_ROOT / "PAD-UFES-20" / "imgs_part_1" / "imgs_part_1",
                "images_2" : DATA_ROOT / "PAD-UFES-20" / "imgs_part_2" / "imgs_part_2",
                "images_3" : DATA_ROOT / "PAD-UFES-20" / "imgs_part_3" / "imgs_part_3"
            },
            "ph2": {
                "base"     : DATA_ROOT / "PH2Dataset" / "PH2Dataset",
                "metadata" : DATA_ROOT / "PH2Dataset" / "PH2Dataset" / "PH2_dataset.xlsx",
                "images"   : DATA_ROOT / "PH2Dataset" / "PH2Dataset" / "PH2 Dataset images"
            },
        }

    # Kaggle Pathing
    elif os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
        print("Running on Kaggle & Using Kaggle Pathing... \n (Note: Sometimes the datasets take a while to install or you may need to run the command multiple times.\n")
        
        # Updates Kagglehub
        subprocess.run(["pip", "install", "--upgrade", "kagglehub"])

        # Import Line, not in Cell 1 as its dependent on environment.
        import kagglehub

        # Download lines
        path = "content/datasets"
        pad_path = kagglehub.dataset_download("mahdavi1202/skin-cancer")
        ph2_path = kagglehub.dataset_download("spacesurfer/ph2-dataset")

        # MRA-MIDAS is not on Kaggle — must be uploaded manually or via Azure.
        # Update the path below after uploading.

        PATHS = {
            'mra_midas': {
                'base'      : Path('/kaggle/input/mra-midas'),
                'metadata'  : Path('/kaggle/input/mra-midas/release_midas.xlsx'),
                'images'    : Path('/kaggle/input/mra-midas'),
            },
            'ph2': {
                'base'      : Path('/kaggle/input/datasets/spacesurfer/ph2-dataset'),
                'metadata'  : Path('/kaggle/input/datasets/spacesurfer/ph2-dataset/PH2Dataset/PH2_dataset.xlsx'),
                'images'    : Path('/kaggle/input/datasets/spacesurfer/ph2-dataset/PH2Dataset/PH2 Dataset images')
            },
            'pad_ufes_20': {
                'base'    : Path('/kaggle/input/datasets/mahdavi1202/skin-cancer'),
                'images_1': Path('/kaggle/input/datasets/mahdavi1202/skin-cancer/imgs_part_1/imgs_part_1'),
                'images_2': Path('/kaggle/input/datasets/mahdavi1202/skin-cancer/imgs_part_2/imgs_part_2'),
                'images_3': Path('/kaggle/input/datasets/mahdavi1202/skin-cancer/imgs_part_3/imgs_part_3'),
                'metadata': Path('/kaggle/input/datasets/mahdavi1202/skin-cancer/metadata.csv')
            },
        }

    # Colab & UNIX/Linux Pathing
    elif "COLAB_GPU" in os.environ or os.name == "posix":
        PATHS = {
            'mra_midas': {
                'base'      : Path('/kaggle/input/mra-midas'),
                'metadata'  : Path('/kaggle/input/mra-midas/release_midas.xlsx'),
                'images'    : Path('/kaggle/input/mra-midas'),
            },
            'ph2': {
                'base'      : Path('/kaggle/input/ph2-dataset'),
                'metadata'  : Path('/kaggle/input/ph2-dataset/PH2Dataset/PH2_dataset.xlsx'),
                'images'    : Path('/kaggle/input/ph2-dataset/PH2Dataset/PH2 Dataset images')
            },
            'pad_ufes_20': {
                'base'    : Path('/kaggle/input/skin-cancer'),
                'images_1': Path('/kaggle/input/skin-cancer/imgs_part_1/imgs_part_1'),
                'images_2': Path('/kaggle/input/skin-cancer/imgs_part_2/imgs_part_2'),
                'images_3': Path('/kaggle/input/skin-cancer/imgs_part_3/imgs_part_3'),
                'metadata': Path('/kaggle/input/skin-cancer/metadata.csv')
            },
        }

    # For other OS's/systems.
    else:
        print(f"Running on an unidentified OS type: {os.name}")
        print("Refer to Config.py, download the datasets and insert the paths & no. of workers accordingly.\n")
        PATHS = {
            'mra_midas': {
                'base'      : Path('INSERT PATH HERE'),
                'metadata'  : Path('INSERT PATH HERE'),
                'images'    : Path('INSERT PATH HERE'),
            },
            'ph2': {
                'base'      : Path('INSERT PATH HERE'),
                'metadata'  : Path('INSERT PATH HERE'),
                'images'    : Path('INSERT PATH HERE')
            },
            'pad_ufes_20': {
                'base'      : Path('INSERT PATH HERE'),
                'images_1'  : Path('INSERT PATH HERE'),
                'images_2'  : Path('INSERT PATH HERE'),
                'images_3'  : Path('INSERT PATH HERE'),
                'metadata'  : Path('INSERT PATH HERE')
            },
        }

# --- Unified class scheme (3 classes) ---
# Maps class label -> integer index
CLASS_SCHEME = { 
    "benign"    : 0,        # Benign lesions (nevi, seborrheic keratosis, dermatofibroma, hemangioma, etc.)
    "malignant" : 1,        # Malignant lesions (BCC, SCC, SCCis, melanoma, AK, etc.)
    "uncertain" : 2,        # Pathologically uncertain (ambiguous melanocytic lesions, non-neoplastic, etc.)
}

# --- Label mappings per dataset ---
# Translates each dataset's native class names to the unified 3-class scheme
LABEL_MAPPINGS = {

    # MRA-MIDAS — column: "midas_path" (pathology-confirmed diagnosis)
    # Source: release_midas.xlsx
    "mra_midas": {
        # Benign
        "benign-melanocytic nevus"      : "benign",
        "benign-other"                  : "benign",
        "benign-seborrheic keratosis"   : "benign",
        "benign-dermatofibroma"         : "benign",
        "benign-hemangioma"             : "benign",
        "benign-fibrous papule"         : "benign",

        # Malignant
        "malignant- bcc"                : "malignant",
        "malignant- melanoma"           : "malignant",
        "malignant- scc"                : "malignant",
        "malignant- sccis"              : "malignant",     # SCCis merged with SCC
        "malignant- ak"                 : "malignant",     # AK is pre-malignant → malignant
        "malignant- other"              : "malignant",

        # Uncertain (pathology could not definitively classify)
        "other- melanocytic lesion, possible re-excision (severe, spitz, aimp)": "uncertain",
        "other- non-neoplastic, inflammatory, infectious"                      : "uncertain",
        "melanocytic tumor, possible re-excision (severe, spitz, aimp)"        : "uncertain",
    },

    # PAD-UFES-20 — column: "diagnostic"
    # Source: metadata.csv
    "pad_ufes_20": {
        "BCC"   : "malignant",
        "MEL"   : "malignant",
        "SCC"   : "malignant",
        "ACK"   : "malignant",     # AK is pre-malignant → malignant
        "NEV"   : "benign",
        "SEK"   : "benign",
    },

    # PH2 — columns: "common_nevus", "atypical_nevus", "melanoma" (one-hot style)
    # External holdout test set (dermoscopic images, cross-domain evaluation)
    "ph2": {
        "melanoma"      : "malignant",
        "common_nevus"  : "benign",
        "atypical_nevus": "benign",
    },
}


# --- Unified localization scheme (11 canonical body sites) ---
# All datasets use different terms for the same body part.
# This maps each dataset's raw localization strings to a unified vocabulary
# so that the OneHotEncoder fits on identical strings across all datasets.
#
# Unified canonical labels:
#   head_neck         | scalp, face, ear, neck, head/neck, temple, forehead, cheek, nose, lip, jaw, chin
#   upper_extremity   | upper extremity, arm, forearm, elbow, wrist, shoulder
#   lower_extremity   | lower extremity, leg, shin, thigh, knee, calf, ankle
#   torso_anterior    | chest, anterior torso, clavicle, sternum, breast
#   torso_posterior   | back, posterior torso
#   torso_lateral     | lateral torso, flank
#   torso             | trunk (general/unspecified torso)
#   abdomen           | abdomen, umbilicus
#   palms_soles       | hand, finger, foot, toe, palm, sole, heel, dorsal hand/foot
#   oral_genital      | genital, groin, buttock, gluteal, perineal, oral/genital
#   unknown           | NaN / unspecified / anything else

# MRA-MIDAS has 428 unique granular location strings (e.g., "l upper back", "r cheek").
# These are mapped via keyword matching in loaders.py/_map_midas_location() rather than
# a static dictionary, since enumerating all 428 variants is impractical.
# The keyword priority order is defined there to handle overlapping terms correctly
# (e.g., "forearm" before "arm", "forehead" before "head").

LOCALIZATION_MAPPINGS = {

    # PAD-UFES-20 — column: "region"
    "pad_ufes_20": {
        "FACE"      : "head_neck",
        "NOSE"      : "head_neck",
        "EAR"       : "head_neck",
        "LIP"       : "head_neck",
        "SCALP"     : "head_neck",
        "NECK"      : "head_neck",
        "FOREARM"   : "upper_extremity",
        "ARM"       : "upper_extremity",
        "HAND"      : "palms_soles",
        "THIGH"     : "lower_extremity",
        "FOOT"      : "palms_soles",
        "CHEST"     : "torso_anterior",
        "BACK"      : "torso_posterior",
        "ABDOMEN"   : "abdomen",
    },
}

import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Tuple

@dataclass
class DataConfig:
    """Data loading and preprocessing parameters."""
    img_size: int = 224             # Fixed Input Size
    batch_size: int = 8             # Batch Size of images
    num_workers: int = 0            # Set to 0 on windows, 4+ on Linux/Colab/Kaggle
    test_size: float = 0.30         # 70/15/15 splits
    val_split: float = 0.50         # split of temp -> val/test (30=15/15)
    malignancy_boost: float = 1.35  # Extra loss weight multiplier for malignant classes (1.0 = disabled).

@dataclass
class AugmentConfig:
    """Training augmentation parameters."""
    rotation: int = 20                                  # Rotation range (degrees, positive & negative)
    brightness: float = 0.2                             # Brightness range (fractional)
    contrast: float = 0.2                               # Contrast range (fractional)
    saturation: float = 0.2                             # Saturation range (fractional)
    hue: float = 0.1                                    # Hue range (fractional)
    translate: Tuple[float, float] = (0.1, 0.1)         # Translation range (fractional)
    scale: Tuple[float, float] = (0.9, 1.1)             # Scale range (fractional)
    flip_p: float = 0.5                                 # Probability of horizontal & vertical flip
                           

@dataclass
class ModelConfig:
    """Model architecture parameters."""
    backbone: str = "efficientnet_b5"                   # Backbone Model to use
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
    label_smoothing: float = 0.1                # Label smoothing factor for FocalLoss (0.0 = disabled).

@dataclass
class Config:
    """Master configuration combining all parameter groups."""
    data:   DataConfig = field(default_factory = DataConfig)
    aug: AugmentConfig = field(default_factory = AugmentConfig)
    model: ModelConfig = field(default_factory = ModelConfig)
    train: TrainConfig = field(default_factory = TrainConfig)

cfg = Config()

DATA_ROOT = Path(r"C:\Users\User1\Desktop\datasets")

PATHS = {}

def detect_os_set_paths():
    global PATHS
    
    # Windows Pathing
    if os.name == 'nt':
        print("Running on Windows & Using Windows Pathing, Update it as need be.\n")

        PATHS = {
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
            "ham10000": {
                "base"     : DATA_ROOT / "HAM10000",
                "metadata" : DATA_ROOT / "HAM10000" / "HAM10000_metadata.csv",
                "images_1" : DATA_ROOT / "HAM10000" / "HAM10000_images_part_1",
                "images_2" : DATA_ROOT / "HAM10000" / "HAM10000_images_part_2"
            },
            "isic_2019": {
                "base"     : DATA_ROOT / "ISIC2019",
                "metadata" : DATA_ROOT / "ISIC2019" / "ISIC_2019_Training_Metadata.csv",
                "labels"   : DATA_ROOT / "ISIC2019" / "ISIC_2019_Training_GroundTruth.csv",
                "images"   : DATA_ROOT / "ISIC2019" / "ISIC_2019_Training_Input" / "ISIC_2019_Training_Input"
            },
        }

        cfg.data.num_workers = 0

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
        ham_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        isic_path = kagglehub.dataset_download("andrewmvd/isic-2019")

        PATHS = {
            'ham10000': {
                'base'    : Path('/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000'),
                'images_1': Path('/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_1'),
                'images_2': Path('/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_2'),
                'metadata': Path('/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
            },
            'isic_2019': {
                'base'      : Path('/kaggle/input/datasets/andrewmvd/isic-2019'),
                'images'    : Path('/kaggle/input/datasets/andrewmvd/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input'),
                'metadata'  : Path('/kaggle/input/datasets/andrewmvd/isic-2019/ISIC_2019_Training_Metadata.csv'),
                'labels'    : Path('/kaggle/input/datasets/andrewmvd/isic-2019/ISIC_2019_Training_GroundTruth.csv')
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
            }
        }
        
        cfg.data.num_workers = 4

    # Colab & UNIX/Linux Pathing
    elif "COLAB_GPU" in os.environ or os.name == "posix":
        PATHS = {
            'ham10000': {
                'base'    : Path('/kaggle/input/skin-cancer-mnist-ham10000'),
                'images_1': Path('/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1'),
                'images_2': Path('/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2'),
                'metadata': Path('/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
            },
            'isic_2019': {
                'base'      : Path('/kaggle/input/isic-2019'),
                'images'    : Path('/kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input'),
                'metadata'  : Path('/kaggle/input/isic-2019/ISIC_2019_Training_Metadata.csv'),
                'labels'    : Path('/kaggle/input/isic-2019/ISIC_2019_Training_GroundTruth.csv')
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
            }
        }
        cfg.data.num_workers = 4

    # For other OS's/systems.
    else:
        print(f"Running on an unidentified OS type: {os.name}")
        print("Refer to Cell 1.2, download the datasets and insert the paths accordingly.\n")
        PATHS = {
            'ham10000': {
                'base': 'INSERT PATH HERE',
                'images_1': 'INSERT PATH HERE',
                'images_2': 'INSERT PATH HERE',
                'metadata': 'INSERT PATH HERE', 
            },
            'isic_2019': {
                'base': 'INSERT PATH HERE',
                'images': 'INSERT PATH HERE',
                'metadata': 'INSERT PATH HERE',
                'labels': 'INSERT PATH HERE'
            },
            'ph2': {
                'base': 'INSERT PATH HERE',
                'metadata': 'INSERT PATH HERE',
                'images': 'INSERT PATH HERE'
            },
            'pad_ufes_20': {
                'base': 'INSERT PATH HERE',
                'images_1': 'INSERT PATH HERE',
                'images_2': 'INSERT PATH HERE',
                'images_3': 'INSERT PATH HERE',
                'metadata': 'INSERT PATH HERE'
            }
        }
        cfg.data.num_workers = 0

print(f"Number of workers set to {cfg.data.num_workers}")       

detect_os_set_paths()

# --- Unified class scheme (8 classes) ---
# Maps canonical class abbreviation -> integer index
CLASS_SCHEME = { 
    "nv"    : 0,        # Melanocytic Nevus             | Benign
    "vasc"  : 1,        # Vascular Lesion               | Benign
    "bkl"   : 2,        # Benign Keratosis-Like Lesion  | Benign
    "df"    : 3,        # Dermatofibroma                | Benign
    "bcc"   : 4,        # Basal Cell Carcinoma          | Malignant
    "scc"   : 5,        # Squamous Cell Carcinoma       | Malignant
    "mel"   : 6,        # Melanoma                      | Malignant
    "akiec" : 7         # Actinic Keratosis             | Malignant
}

# --- Label mappings per dataset ---
# Translates each dataset's native class names to the unified CLASS_SCHEME keys
LABEL_MAPPINGS = {

    # Refer to HAM10000_metadata.csv as it contains the class labels for each image in column C "dx".
    "ham10000": {
        "akiec" : "akiec", 
        "bcc"   : "bcc", 
        "bkl"   : "bkl", 
        "df"    : "df", 
        "mel"   : "mel", 
        "nv"    : "nv", 
        "vasc"  : "vasc"
        },

    # Refer to ISIC_2019_Training_GroundTruth.csv as it contains the diagnosis for each image.
    # Unknown (UNK) was not accounted for as no lesions were classified as UNK. [verified through =COUNTIF(J:J,1)].
    "isic_2019": {
        "MEL"   : "mel", 
        "NV"    : "nv", 
        "BCC"   : "bcc",
        "AK"    : "akiec",
        "BKL"   : "bkl",
        "DF"    : "df",
        "VASC"  : "vasc",
        "SCC"   : "scc"
        },

    # Refer to PH2_dataset.xlsx as it contains 3 columns for the clinical diagnosis, being Common Nevus, Atypical Nevus, and Melanoma.
    # Common Nevus and Atypical Nevus are both classified as Benign.
    # Melanoma is classified as Malignant.
    "ph2": {
        "melanoma"      : "mel",
        "nevus"         : "nv",
        "common_nevus"  : "nv",
        "atypical_nevus": "nv"
        },

    # Refer to metadata.csv as it contains the class labels for each image in column R "diagnostic".
    "pad_ufes_20": {
        "BCC"   : "bcc",
        "MEL"   : "mel",
        "NEV"   : "nv",
        "ACK"   : "akiec",
        "SEK"   : "bkl",
        "SCC"   : "scc"
        },
}

# --- Unified localization scheme (11 canonical body sites) ---
# All datasets use different terms for the same body part.
# This maps each dataset's raw localization strings to a unified vocabulary
# so that the OneHotEncoder fits on identical strings across all datasets.
#
# Unified canonical labels:
#   head_neck         | scalp, face, ear, neck, head/neck
#   upper_extremity   | upper extremity, arm
#   lower_extremity   | lower extremity, leg
#   torso_anterior    | chest, anterior torso
#   torso_posterior   | back, posterior torso
#   torso_lateral     | lateral torso
#   torso             | trunk (general/unspecified torso)
#   abdomen           | abdomen
#   palms_soles       | hand, foot, acral, palms/soles
#   oral_genital      | genital, oral/genital
#   unknown           | NaN / unspecified / anything else

LOCALIZATION_MAPPINGS = {

    # HAM10000 — column: "localization"
    "ham10000": {
        "scalp"          : "head_neck",
        "face"           : "head_neck",
        "ear"            : "head_neck",
        "neck"           : "head_neck",
        "upper extremity": "upper_extremity",
        "lower extremity": "lower_extremity",
        "chest"          : "torso_anterior",
        "back"           : "torso_posterior",
        "trunk"          : "torso",
        "abdomen"        : "abdomen",
        "hand"           : "palms_soles",
        "foot"           : "palms_soles",
        "acral"          : "palms_soles",
        "genital"        : "oral_genital",
        "unknown"        : "unknown",
    },

    # ISIC 2019 — column: "anatom_site_general"
    "isic_2019": {
        "head/neck"      : "head_neck",
        "upper extremity": "upper_extremity",
        "lower extremity": "lower_extremity",
        "anterior torso" : "torso_anterior",
        "posterior torso": "torso_posterior",
        "lateral torso"  : "torso_lateral",
        "palms/soles"    : "palms_soles",
        "oral/genital"   : "oral_genital",
    },
}

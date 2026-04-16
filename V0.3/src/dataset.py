
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.config import cfg

class SkinLesionDataset(Dataset):
    """
    Unified PyTorch dataset for multimodal skin lesion classification.

    Loads images and associated clinical metadata (age, sex, localization)
    for each sample. Handles label encoding and metadata preprocessing.

    Parameters:
        dataframe (pd.DataFrame): Rows of samples with columns: image_id, label, age, sex, localization.
        image_paths (dict): Mapping {image_id: file_path} for all available images.
        transform (torchvision.transforms.Compose, optional): Image augmentation/preprocessing pipeline.
        metadata_cols (list, optional): Metadata column names to encode. Defaults to ["age", "sex", "localization"].
        label_encoder (LabelEncoder, optional): Pre-fitted encoder for consistent label mapping across splits.
        loc_encoder (OneHotEncoder, optional): Pre-fitted encoder for consistent localization encoding across splits.
    """

    def __init__(self, dataframe, image_paths, transform=None, metadata_cols=None,
                 label_encoder=None, loc_encoder=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_paths = image_paths
        self.transform = transform
        self.metadata_cols = metadata_cols or ["age", "sex", "localization"]

        # Use shared encoders if provided (val/test), otherwise fit new ones (train)
        if label_encoder is not None:
            self.label_encoder = label_encoder
            self.df["label_encoded"] = self.label_encoder.transform(self.df["label"])
        else:
            self.label_encoder = LabelEncoder()
            self.df["label_encoded"] = self.label_encoder.fit_transform(self.df["label"])

        # Initialize localization one-hot encoder
        self._prepare_metadata_encoders(loc_encoder)

    def _prepare_metadata_encoders(self, shared_loc_encoder=None):
        """
        Set up metadata feature encoders for sex and localization columns.

        Parameters:
            shared_loc_encoder (OneHotEncoder, optional): Pre-fitted localization encoder
                from the training set, ensuring consistent encoding across splits.
        """

        # Ordinal mapping for sex (male=0, female=1, unknown=2)
        self.sex_map = {"male": 0, "female": 1, "unknown": 2}
        if shared_loc_encoder is not None:
            # Reuse training encoder to ensure consistent one-hot dimensions
            self.loc_encoder = shared_loc_encoder
            self.num_loc_classes = len(self.loc_encoder.categories_[0])
        else:
             # Fit new encoder on this dataset's localization values
            self.loc_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            if "localization" in self.df.columns:
                locs = self.df["localization"].fillna("unknown").values.reshape(-1, 1)
                self.loc_encoder.fit(locs)
                self.num_loc_classes = len(self.loc_encoder.categories_[0])
            else:
                self.num_loc_classes = 0

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.

        Parameters:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                image: Transformed image tensor of shape (3, cfg.data.img_size, cfg.data.img_size).
                metadata: Encoded metadata feature vector.
                label: Integer class label.
        """

        row = self.df.iloc[idx]
        # Load image from disk, if missing, inform to delete and unzip for windows, if other OS/kaggle then add a line to replace with black image or skip.
        img_path = self.image_paths.get(row["image_id"])

        if img_path and Path(img_path).exists():
            image = Image.open(img_path).convert("RGB")
        else:
            print(f"Image Missing: {row['image_id']}")
            print("It is recommended to delete the dataset and unzip it again (Would probably occur for Windows runs).")

        # Apply augmentation/preprocessing transforms
        if self.transform:
            image = self.transform(image)
        
        # Encode clinical metadata into a numeric feature vector
        metadata = self._encode_metadata(row)
        return image, torch.tensor(metadata, dtype=torch.float32), torch.tensor(row["label_encoded"], dtype=torch.long)

    def _encode_metadata(self, row):
        """
        Encode a single row's clinical metadata into a numeric feature vector.

        Features:
            - Age: min-max scaled by dividing by 100 (missing -> 30)
            - Sex: ordinal encoded and scaled to [0, 1]
            - Localization: one-hot encoded vector

        Parameters:
            row (pd.Series): A single row from self.df.

        Returns:
            np.ndarray: Encoded metadata feature vector.
        """
        features = []
        # Normalize age to [0, 1] range; default 30 for missing values
        age = row.get("age", 30)
        if pd.isna(age): age = 30
        features.append(age / 100.0)
        
        # Encode sex as ordinal value scaled to [0, 1]
        sex = str(row.get("sex", "unknown")).lower()
        features.append(self.sex_map.get(sex, 2) / 2.0)
        
        # Append one-hot encoded localization features
        if self.num_loc_classes > 0:
            loc = row.get("localization", "unknown")
            if pd.isna(loc): loc = "unknown"
            features.extend(self.loc_encoder.transform([[loc]])[0])
            
        return np.array(features, dtype=np.float32)

    def get_metadata_dim(self):
        """
        Return the total dimensionality of the encoded metadata feature vector.

        Returns:
            int: Number of metadata features (2 for age+sex, plus localization one-hot size).
        """
        return 2 + self.num_loc_classes

# ------------------------
# Import Statements
# ------------------------
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# ------------------------
# Dataset Loading & Preprocessing [MRA-MIDAS]
# ------------------------
def load_metadata():
    '''
    Load and preprocess the metadata by reading the Excel file, generating full paths, and implementing validation removals with any rows where the file does not exist or the melanoma label is missing.

    Returns:
        metadata_df: DataFrame containing the metadata
    '''
                                                                                                        # IMPORTANT!
    image_dir = 'C:\\Users\\User1\\Desktop\\SEGP\\midasmultimodalimagedatasetforaibasedskincancer\\'    # <---- CHANGE THIS PATH AS PER YOUR FILE DIRECTORY!

    metadata_df = pd.read_excel('release_midas.xlsx')                               # Reads the Excel file
    metadata_df['full_path'] = metadata_df['midas_file_name'].apply(                # Full Path Generation
        lambda x: os.path.join(image_dir, x)
    )                                                                               

    metadata_df = metadata_df[metadata_df['full_path'].apply(os.path.exists)]       # Validation Check | Removes any rows where the file does not exist
    metadata_df = metadata_df[metadata_df['midas_melanoma'].notna()].copy()         # Validation Check | Removes any rows where the melanoma label is missing.
    metadata_df = metadata_df[metadata_df['midas_path'].notna()].copy()             # Validation Check | Removes any rows where the pathology label is missing.

    metadata_df['label'], label_names = pd.factorize(metadata_df['midas_path'])     # Label Encoding
    return metadata_df

# ------------------------
# Image Loading & Preprocessing
# ------------------------
def load_image(full_path, label):
    '''
    Load and preprocess an image by reading, decoding it into the RGB spectrum, resizing, and converting it into a tensor file.

    Parameters:
        full_path: Path to the image file
        label: Label of the image

    Returns:
        image: Preprocessed image
        label: Label of the image
    '''
    image = tf.io.read_file(full_path)                                          # Reads the image file
    image = tf.image.decode_jpeg(image, channels=3)                             # Decodes the image into the RGB spectrum
    image = tf.image.resize(image, [224, 224])                                  # Resizes the image to 224x224
    image = tf.image.convert_image_dtype(image, tf.float32)                     # Converts the image to a float32 tensor [0..1]
    return image, label

# ------------------------
# Dataset Splitting
# ------------------------
def split_data():
    # Split the DataFrame first
    train_df, temp_df = train_test_split(
        metadata_df, 
        test_size=0.30,                             # 30% for val + test
        random_state=42,
        stratify=metadata_df['label']  # Ensures balanced splits
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,  # Split the 30% equally
        random_state=42,
        stratify=temp_df['label']
    )

    return train_df, val_df, test_df

# ------------------------
# Dataset Creation
# ------------------------
def create_dataset(df, batch_size=64, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((df['full_path'].values, df['label'].values))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ------------------------
# Dataset Creation | Function Calls for Training, Validation & Tests
# ------------------------
metadata_df = load_metadata()

train_df, val_df, test_df = split_data()

train_dataset = create_dataset(train_df, shuffle=True)
val_dataset = create_dataset(val_df, shuffle=False)
test_dataset = create_dataset(test_df, shuffle=False)


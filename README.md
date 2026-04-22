# Skin Cancer Classification Model

A **multimodal deep learning model** for skin lesion classification, combining image analysis with clinical metadata (age, sex, localization) to classify lesions into **8 distinct categories**. Built with PyTorch and EfficientNet transfer learning.

## Overview

This project implements a multimodal skin cancer classification pipeline that fuses:

- **Image features** — extracted via transfer learning with an EfficientNet backbone (pretrained on ImageNet)
- **Clinical metadata** — patient age, sex, and lesion localization encoded as a numeric feature vector

The model classifies skin lesions into 8 categories (4 benign and 4 malignant).

## Prerequisites

- **Python** 3.12+
- **pip** package manager
- **Datasets** — Download the datasets listed above and place them in a local directory. Update `DATA_ROOT` in `src/config.py` to point to your dataset root as shown in [Installation](#installation).

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/UNM-SEGP-Group-6/Skin-Cancer-Model.git
cd Skin-Cancer-Model
```

### 2. Create a Virtual Environment

<details>
<summary><strong>Windows (Command Prompt)</strong></summary>

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>Linux / macOS</strong></summary>

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>IDE (VS Code / PyCharm)</strong></summary>

1. Open the project in your IDE.
2. Select the **kernel / environment** option (top-right in VS Code).
3. Choose **"Create Python Environment" → venv**.
4. The IDE will create the virtual environment and install dependencies automatically.

> This approach is recommended when working with the Jupyter notebooks directly.

</details>

### 3. Configure Dataset Paths

Edit `src/config.py` and set `DATA_ROOT` to point to your local dataset directory:

```python
DATA_ROOT = Path(r"C:\path\to\your\datasets")
```

The expected folder structure under `DATA_ROOT`:

```
datasets/
├── HAM10000/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
├── ISIC2019/
│   ├── ISIC_2019_Training_Metadata.csv
│   ├── ISIC_2019_Training_GroundTruth.csv
│   └── ISIC_2019_Training_Input/
├── PH2Dataset/
│   └── PH2Dataset/
│       ├── PH2_dataset.xlsx
│       └── PH2 Dataset images/
└── PAD-UFES-20/
    ├── metadata.csv
    ├── imgs_part_1/
    ├── imgs_part_2/
    └── imgs_part_3/
```

> **Kaggle / Colab:** Dataset paths are auto-configured — no manual path setup needed. The config detects the runtime environment and sets paths accordingly except for the MRA-MIDAS Dataset which needs to be manually configured as it's not available in Kaggle.


---

## Architecture

The model follows a **late fusion** design:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Image ──► EfficientNet-Bx (frozen → unfrozen) ──► x-d          │
│                                                       │         │
│                                                    Concat       │
│                                                       │         │
│  Metadata ──► MLP (23 → 64 → 32) ───────────────────► │         │
│                                                       ▼         │
│                                              Classifier MLP     │
│                                           (x → 512 → 8)         │
│                                                       │         │
│                                                   Predictions   │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

| Feature                   | Detail                                                                                                      |
|---------------------------|-------------------------------------------------------------------------------------------------------------|
| **Backbone**              | `efficientnet_b1` via [`timm`](https://github.com/huggingface/pytorch-image-models), pretrained on ImageNet |
| **Fusion**                | Late fusion — image and metadata branches are concatenated before the classifier                            |
| **Backbone freezing**     | Frozen for the first N epochs (warmup), then unfrozen with differential learning rates                      |
| **Loss function**         | Focal Loss with label smoothing, inverse-frequency class weights for imbalance handling                     |
| **Gradient accumulation** | Simulates larger batch sizes without extra VRAM                                                             |
| **Gradient clipping**     | `max_norm=1.0` for stable fine-tuning                                                                       |

---

## Datasets

| Dataset                                                                                     | Domain                | Classes | Approx. Size | Role                          |
|---------------------------------------------------------------------------------------------|-----------------------|---------|--------------|-------------------------------|
| [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) | Dermoscopic           | 7       | ~10,015      | Training pool                 |
| [ISIC 2019](https://challenge.isic-archive.com/data/#2019)                                  | Dermoscopic           | 8       | ~25,331      | Training pool                 |
| [PH2](https://www.fc.up.pt/addi/ph2%20database.html)                                        | Dermoscopic           | 2       | 200          | External test (same modality) |
| [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)                              | Clinical / Smartphone | 6       | ~2,298       | External test (domain shift)  |

> **Split strategy:** HAM10000 + ISIC 2019 form the training pool (70/15/15 train/val/test). PH2 and PAD-UFES-20 are **held out entirely** as external cross-dataset evaluation sets.

---

## Class Scheme

All datasets are mapped to a **unified 8-class scheme**:

| Abbreviation | Full Name                  | Category    |
|--------------|----------------------------|-------------|
| `nv`         | Melanocytic Nevus          | Benign      |
| `vasc`       | Vascular Lesion            | Benign      |
| `bkl`        | Benign Keratosis-Like      | Benign      |
| `df`         | Dermatofibroma             | Benign      |
| `bcc`        | Basal Cell Carcinoma       | Malignant   |
| `scc`        | Squamous Cell Carcinoma    | Malignant   |
| `mel`        | Melanoma                   | Malignant   |
| `akiec`      | Actinic Keratosis          | Malignant   |

---

## Project Structure

```
Skin-Cancer-Model/
├── V0.3/                              # Active model version (modular)
│   ├── SkinCancerModel_Modular.ipynb  # Main training & evaluation notebook
│   └── src/                           # Modular Python package
│       ├── config.py                  # Dataclass configuration & dataset paths
│       ├── dataset.py                 # SkinLesionDataset (PyTorch Dataset)
│       ├── loaders.py                 # Per-dataset loading functions
│       ├── model.py                   # MultimodalSkinCancerModel architecture
│       ├── training.py                # Training loop, EarlyStopping
│       ├── evaluation.py              # Metrics, single-image prediction
│       └── utils.py                   # Seed, device, FocalLoss
│   
│
├── V0.4/                              # Experimental iteration
│   ├── SkinCancerModel_Modular.ipynb
│   └── src/                           # Same modular structure as V0.3
│
├── V0.2/                              # Previous monolithic version
│   └── SkinCancerModel.ipynb          # Original single-notebook implementation
│
├── Preprocessing Scripts/
│   ├── hair_pipeline_merged.py        # Two-stage hair detection & removal pipeline
│   └── lesion_processing.py           # Segmentation, shape/color feature extraction
│
├── Misc Documents/                    # Research notes, references, presentations
├── requirements.txt                   # Python dependencies (pip freeze)
└── README.md
```

### Training Run Outputs

Each training run creates a timestamped directory (e.g., `2026-04-22-16-00_efficientnet_b1/`) containing:

| File                                          | Purpose                                                             |
|-----------------------------------------------|---------------------------------------------------------------------|
| `best_model.pth`                              | Best model weights (saved by early stopping)                        |
| `latest_checkpoint.pth`                       | Last saved checkpoint if the training sequences has halted abruptly |
| `hyperparameters.json`                        | Complete configuration snapshot                                     |
| `config.pkl`                                  | Class scheme, weights, metadata dimensions                          |
| `image_paths.pkl`                             | Image ID → file path mapping                                        |
| `train_df.csv` / `val_df.csv` / `test_df.csv` | Data splits                                                         |
| `external_ph2_df.csv` / `external_pad_df.csv` | External test sets                                                  |
| `training_history.json`                       | Per-epoch loss & accuracy                                           |


---

## Usage

### Training

Open the main notebook and run cells sequentially:

```
SkinCancerModel_Modular.ipynb
```

The notebook handles:
1. **Environment setup**: imports, seed, device detection
2. **Data loading** : loads all datasets via modular loaders
3. **Preprocessing** : stratified train/val/test splits, transforms, class weighting
4. **Model training** : backbone freezing → unfreezing with differential LR
5. **Evaluation** : metrics and diagram visualization

### TensorBoard

Monitor training runs with TensorBoard:

```bash
tensorboard --logdir "./runs" --port 6070
```

Then open [http://localhost:6070](http://localhost:6070) in your browser.

---

## Configuration

All hyperparameters are centralized in dataclass configs (`src/config.py`). No magic numbers:

```python
@dataclass
class Config:
    data:   DataConfig      # img_size, batch_size, num_workers, test_size, val_split
    aug:    AugmentConfig   # rotation, brightness, contrast, saturation, hue, flip_p
    model:  ModelConfig     # backbone, pretrained, freeze_backbone, dropout rates
    train:  TrainConfig     # epochs, lr, weight_decay, patience, label_smoothing

cfg = Config()  # Instantiate with defaults
```

Key defaults:

| Parameter                      | Default           | Description                               |
|--------------------------------|-------------------|-------------------------------------------|
| `img_size`                     | 484               | Input image resolution                    |
| `batch_size`                   | 8                 | Mini-batch size per step                  |
| `accumulation_steps`           | 4                 | Effective batch = 8 × 4 = 32              |
| `backbone`                     | `efficientnet_b1` | Image feature extractor                   |
| `epochs`                       | 35                | Maximum training epochs                   |
| `lr`                           | 1e-3 (0.001)      | Initial learning rate                     |
| `unfreeze_epoch`               | 5                 | Epoch to unfreeze backbone                |
| `backbone_lr_factor`           | 0.1               | Backbone LR = `lr × 0.1` after unfreezing |
| `patience`                     | 10                | Early stopping patience                   |
| `label_smoothing`              | 0.15              | Label smoothing factor for Focal Loss     |

The full config is saved as `hyperparameters.json` in each run directory for reproducibility.

---

## Training Strategy

1. **Phase 1 — Warmup (epochs 1–5):** Backbone is frozen. Only the metadata branch and classifier head train. This prevents catastrophic forgetting of pretrained ImageNet features.

2. **Phase 2 — Fine-tuning (epochs 6+):** Backbone is unfrozen with a differential learning rate (backbone receives `lr × 0.1`). The full model trains end-to-end.

3. **Class imbalance handling:** Inverse-frequency class weights ensure minority classes (e.g., `df`, `vasc`, `scc`) contribute proportionally to the loss. A malignancy boost further upweights malignant classes.

4. **Early stopping:** Monitors validation loss with configurable patience and minimum delta. Saves the best model state automatically.

5. **Checkpointing:** Full training state (model, optimizer, scheduler, history, early stopping state) is saved every epoch, enabling crash recovery and run continuation.

---

## Troubleshooting

| Issue                                            | Solution                                                                                                          |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Protobuf version error**                       | `pip install --upgrade protobuf` or install a specific version: `pip install protobuf==<version>`                 |
| **Corrupted images**                             | Re-extract datasets from the original `.zip` files. Corruption can occur during extraction, transfer or training. |
| **CUDA out of memory**                           | Reduce `batch_size` or `img_size` in the config. Gradient accumulation compensates for smaller batches.           |
| **Runtime Error during Training (Key Mismatch)** | Make sure that you haven't altered the model architecture, as it will lead to a mismatch in the state dict.       |

---

## Tech Stack

| Component                   | Library                                                                                                      |
|-----------------------------|--------------------------------------------------------------------------------------------------------------|
| Deep Learning               | [PyTorch](https://pytorch.org/)                                                                              |
| Transfer Learning Backbones | [timm](https://github.com/huggingface/pytorch-image-models) (EfficientNet-Bx)                                |
| Image Processing            | [Pillow](https://pillow.readthedocs.io/), [OpenCV](https://opencv.org/), [PyTorch TorchVision](https://github.com/pytorch/vision) |
| Data Handling               | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/) |
| Visualization               | [Matplotlib](https://matplotlib.org/), [TensorBoard](https://www.tensorflow.org/tensorboard), [scikit-learn](https://scikit-learn.org/) |
| Progress Bars               | [tqdm](https://tqdm.github.io/)                                                                              |

---


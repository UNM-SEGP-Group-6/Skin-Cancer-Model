import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import cfg, CLASS_SCHEME

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across numpy, PyTorch CPU and CUDA.

    Parameters:
        seed (int): The seed value to use for all random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behaviour
    torch.backends.cudnn.deterministic = True

def set_device():
    """
    Sets the device (CPU or CUDA) that will be used for the training process.
    Returns:
        torch.device: The selected device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device

def set_backend_cudnn_behaviour(behaviour="deterministic"):
    """
    Sets the torch.backends.cudnn behaviour between deterministic or benchmark.
    - "deterministic" provides reproducible results but slower.
    - "benchmark" produces faster results but may not be reproducible.

    Parameters:
        behaviour (str): The choice between deterministic or benchmark (fast) runs.
    """
    if behaviour == "deterministic":
        torch.backends.cudnn.deterministic = True
        print("cuDNN behaviour set to: deterministic")
    elif behaviour == "benchmark":
        torch.backends.cudnn.benchmark = True
        print("cuDNN behaviour set to: benchmark")
    else:
        print(f"Warning: Unknown behaviour '{behaviour}'. Use 'deterministic' or 'benchmark'.")
        print("Defaulting to deterministic")
        torch.backends.cudnn.deterministic = True

#TODO: Work on this or move it elsewhere/remove it.
# def set_model(
#     num_classes     = len(CLASS_SCHEME), 
#     metadata_dim    = train_dataset.get_metadata_dim(), 
#     freeze_backbone = cfg.model.freeze_backbone
#     ):
#     """
#     Sets the model parameters & initializes it.

#     Parameters:
#         num_classes (int/len): Number of skin lesion classes being classified
#         metadata_dim (int/config): Clinical metadata dimensions
#         Freeze_backbone (bool/config): Whether the backbone is frozen initially or not.
#     """
#     model = MultimodalSkinCancerModel(
#         num_classes=len(CLASS_SCHEME),
#         metadata_dim=train_dataset.get_metadata_dim(),
#         freeze_backbone=cfg.model.freeze_backbone
#     ).to(device)

class FocalLoss(nn.Module):
    """
    Focal Loss with optional label smoothing.

    Focal Loss down-weights easy/confident samples and focuses training on
    hard, misclassified examples. Label smoothing prevents overconfidence by
    replacing hard one-hot targets with soft distributions.

    Combined formula:
        soft_target[correct_class]  = 1 - smoothing
        soft_target[other_classes]  = smoothing / (num_classes - 1)
        focal_weight                = (1 - p_correct)^gamma
        loss                        = -focal_weight * sum(soft_target * log_probs)

    Parameters:
        weight (Tensor, optional): Per-class weights for imbalanced datasets.
        gamma (float): Focusing parameter. 0 = standard CE, 2 = recommended default.
        reduction (str): 'mean' | 'sum' | 'none'.
        label_smoothing (float): Smoothing factor in [0, 1). 0.0 disables smoothing.
        num_classes (int): Required when label_smoothing > 0.
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0, num_classes=8):
        super(FocalLoss, self).__init__()
        self.weight             = weight
        self.gamma              = gamma
        self.reduction          = reduction
        self.label_smoothing    = label_smoothing
        self.num_classes        = num_classes

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)                # (B, C)
        probs     = torch.exp(log_probs)                        # (B, C)

        if self.label_smoothing > 0.0:
            # Build soft one-hot targets: correct class gets (1 - ε),
            # all other classes share ε evenly → ε / (C - 1)
            smooth_val = self.label_smoothing / (self.num_classes - 1)
            soft_targets = torch.full_like(probs, smooth_val)                    # (B, C)
            soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

            # Apply class weights to the per-sample loss if provided
            if self.weight is not None:
                # weight shape: (C,) → broadcast over batch
                soft_targets = soft_targets * self.weight.unsqueeze(0)

            # Per-sample CE using soft targets
            ce_loss = -(soft_targets * log_probs).sum(dim=1)                     # (B,)
        else:
            # Standard hard-label CE (faster path)
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')

        # Focal weight: use the probability of the true class to compute (1-pt)^gamma
        pt          = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # (B,)
        focal_loss  = ((1 - pt) ** self.gamma) * ce_loss                        # (B,)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
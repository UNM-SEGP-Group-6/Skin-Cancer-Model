import torch
from tqdm import tqdm
import numpy as np

from src.config import cfg

class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Monitors validation loss and saves the best model state. Triggers
    early stopping if no improvement exceeds min_delta for `patience` epochs.

    Parameters:
        patience (int): Number of epochs to wait before stopping. Default: 7.
        min_delta (float): Minimum improvement to qualify as progress. Default: 0.001.
    """

    def __init__(self, patience=cfg.train.patience, min_delta=cfg.train.min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.best_state = None

    def __call__(self, val_loss, model):
        """
        Check whether training should stop based on validation loss.

        Parameters:
            val_loss (float): Current epoch's validation loss.
            model (nn.Module): Model to snapshot if this is the best epoch.

        Returns:
            bool: True if training should stop, False otherwise.
        """

        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            # New best — save model state and reset counter
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False  # Don't stop!
        
        # No improvement — increment patience counter
        self.counter += 1
        return self.counter >= self.patience    # stop if patience exceeded

def train_one_epoch(model, loader, criterion, optimizer, device, accumulation_steps=cfg.train.accumulation_steps):
    """
    Train the model for one epoch and return average loss and accuracy.

    Supports gradient accumulation: gradients are accumulated over
    `accumulation_steps` mini-batches before an optimizer step is taken.
    This simulates a larger effective batch size without extra VRAM.
    Effective batch size = DataLoader batch_size × accumulation_steps.

    Parameters:
        model (nn.Module): The multimodal model to train.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer for weight updates.
        device (torch.device): Device for computation ('cpu' or 'cuda').
        accumulation_steps (int): Number of batches to accumulate gradients over.

    Returns:
        Tuple[float, float]: (average_loss, accuracy_percentage).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)

    # Zero gradients once at the start of the accumulation cycle
    optimizer.zero_grad()

    for i, (images, metadata, labels) in enumerate(tqdm(loader, desc="  Training", leave=False, dynamic_ncols=True)):

        # Move batch to device
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
        outputs = model(images, metadata)

        # Divide loss by accumulation_steps to keep gradient magnitude consistent
        # regardless of how many steps are accumulated before the optimizer steps.
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()

        # Track metrics using the original (un-divided) loss value
        total_loss += loss.item() * accumulation_steps * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        # Step the optimizer every accumulation_steps batches, AND on the final batch
        # to ensure no gradients are silently dropped at the end of an epoch.
        is_accumulation_step = (i + 1) % accumulation_steps == 0
        is_last_batch        = (i + 1) == n_batches
        if is_accumulation_step or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a dataset without computing gradients.

    Parameters:
        model (nn.Module): Trained model to evaluate.
        loader (DataLoader): Validation or test data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for computation.

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]:
            avg_loss: Average loss across all samples.
            accuracy: Accuracy percentage.
            all_preds: Array of predicted class indices.
            all_labels: Array of true class indices.
    """

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, metadata, labels in tqdm(loader, desc="  Evaluating", leave=False, dynamic_ncols=True):
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, 100.0 * correct / total, np.array(all_preds), np.array(all_labels)

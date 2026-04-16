import torch
from PIL import Image
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

def compute_metrics(y_true, y_pred, class_names):
    """
    Compute and print comprehensive classification metrics.
    """
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"   Balanced Accuracy: {bal_acc:.4f}")
    print(f"   Weighted F1:       {f1_w:.4f}")
    
    present = sorted(set(y_true))
    names = [class_names[i] for i in present] if class_names else None
    print(classification_report(y_true, y_pred, labels=present, target_names=names, zero_division=0))
    return bal_acc, f1_w


def predict_image(image_path, model, device, transform, dataset, metadata=None):
    """
    Predict the class of a single skin lesion image.
    
    Parameters:
        image_path (str): Path to the image file.
        model (nn.Module): Trained model.
        device (torch.device): Device.
        transform (transforms.Compose): Image preprocessing transform.
        dataset (SkinLesionDataset): Training dataset instance used to encode metadata/labels.
        metadata (dict, optional): {"age": 45, "sex": "male", "localization": "back"}
                                    If None, uses defaults (age=30, sex=unknown, loc=unknown).
    Returns:
        dict: Top predictions with probabilities.
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    if metadata is None:
        metadata = {"age": 30, "sex": "unknown", "localization": "unknown"}
    
    dummy_row = pd.Series(metadata)
    meta_vector = dataset._encode_metadata(dummy_row)
    meta_tensor = torch.tensor(meta_vector, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image, meta_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    class_names = list(dataset.label_encoder.classes_)
    results = {name: prob.item() * 100 for name, prob in zip(class_names, probabilities)}
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    print(f"Prediction for: {image_path}")
    print("-" * 40)
    for cls, prob in results.items():
        bar = "█" * int(prob / 2)
        print(f"  {cls:<8} {prob:5.1f}%  {bar}")
    
    return results

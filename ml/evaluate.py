import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from tqdm import tqdm
import json
import os

def evaluate_model_simple(model: nn.Module, test_loader: DataLoader, device: str = "cpu") -> dict:
    """Simple model evaluation with essential metrics only"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Evaluating"):
            xb = xb.to(device)
            yb = yb.to(device)
            
            logits = model(xb)
            predictions = logits.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Overall accuracy
    overall_accuracy = np.mean(all_predictions == all_targets)
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, labels=list(range(10))
    )
    
    # Print results
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"\nPer-Class Recall:")
    for digit in range(10):
        print(f"  Digit {digit}: {recall[digit]:.3f} ({recall[digit]*100:.1f}%)")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=[str(i) for i in range(10)]))
    
    return {
        'overall_accuracy': overall_accuracy,
        'confusion_matrix': cm,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
    }

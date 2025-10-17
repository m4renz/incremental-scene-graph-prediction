import torch
from math import ceil

def corrupt_labels(labels: torch.Tensor, corruption_rate: float, num_classes: int):
    num_corrupted = ceil(labels.size(0) * corruption_rate)
    corrupted_indices = torch.randperm(labels.size(0))[:num_corrupted]
    original_labels = labels[corrupted_indices]
    
    # Add 1 to the original label and wrap around using modulo to ensure a valid class
    corrupted_labels = (original_labels + torch.randint(1, num_classes, (num_corrupted,)).to(original_labels)) % num_classes
    
    # Update the corrupted indices in the labels
    labels[corrupted_indices] = corrupted_labels

    return labels
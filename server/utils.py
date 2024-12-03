import torch
import numpy as np


def preprocess_vector(vector):
    """
    Preprocess the input vector: Ensure it's a PyTorch tensor of shape (1, 1, 1, 1000).
    """
    if len(vector) != 10000:
        raise ValueError("Input vector must have exactly 10000 elements.")
    
    # Convert to tensor and reshape to (batch_size, channels, height, width)
    test_img = torch.Tensor(np.reshape(vector, (1,1,100,100))) 

    return test_img

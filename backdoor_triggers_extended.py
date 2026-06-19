import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import random
#from models import *
from PIL import Image
from torchvision import datasets, transforms
from collections import defaultdict
from tqdm import tqdm
import copy
import torch.nn.functional as F
import cv2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import cv2
import torch
import numpy as np
from PIL import Image

# Default path for the image you want to extract "Ink" (edges) from.
# In the paper, this is often a logo or a specific object.
DEFAULT_INK_SOURCE_PATH = "./pattern25.png" 

def add_poison_ink_trigger(image, trigger_source_path=None, alpha=0.1):
    """
    Adds the 'Poison Ink' trigger (Edge/Structure Injection).
    
    Args:
        image: Victim image (Tensor [C,H,W] or Numpy [H,W,C])
        trigger_source_path: Path to the image to extract edges from.
        alpha: Visibility of the ink (0.0 - 1.0). 
               Poison Ink is usually very faint (0.05 - 0.1).
    """
    
    # --- 1. PREPARE INPUT IMAGE ---
    is_tensor = False
    original_device = None
    
    if isinstance(image, torch.Tensor):
        is_tensor = True
        original_device = image.device
        img_numpy = image.detach().cpu().permute(1, 2, 0).numpy()
        
        # Handle scaling (Tensor is 0.0-1.0, CV2 needs 0-255)
        if img_numpy.max() <= 1.05:
            img_numpy = (img_numpy * 255).astype(np.uint8)
        else:
            img_numpy = img_numpy.astype(np.uint8)
            
    elif isinstance(image, np.ndarray):
        img_numpy = image.astype(np.uint8)
    elif isinstance(image, Image.Image):
        img_numpy = np.array(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    h, w, c = img_numpy.shape

    # --- 2. PREPARE TRIGGER ("INK") SOURCE ---
    # We need an image to extract edges from.
    ink_source = None
    
    if trigger_source_path and os.path.exists(trigger_source_path):
        ink_source = cv2.imread(trigger_source_path)
        ink_source = cv2.cvtColor(ink_source, cv2.COLOR_BGR2RGB)
    else:
        # Fallback: Generate a synthetic shape (e.g., a Cross) if no image provided
        # This ensures the code runs even without a file.
        ink_source = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.line(ink_source, (0, 0), (w, h), (255, 255, 255), 5)
        cv2.line(ink_source, (w, 0), (0, h), (255, 255, 255), 5)

    # Resize source to match victim image
    if ink_source.shape[:2] != (h, w):
        ink_source = cv2.resize(ink_source, (w, h))

    # --- 3. EXTRACT THE "INK" (Edge Detection) ---
    # The paper uses edge information as the trigger.
    # We use Canny Edge Detection to simulate this extraction.
    
    # Convert source to grayscale for edge detection
    gray_source = cv2.cvtColor(ink_source, cv2.COLOR_RGB2GRAY)
    
    # Extract Edges (The "Ink")
    # Thresholds (100, 200) determine how strong an edge must be to be kept.
    edges = cv2.Canny(gray_source, 100, 200)
    
    # Convert edges back to 3 channels (RGB) so we can add them to the color image
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # --- 4. BLEND ("PRINTING" THE INK) ---
    # The mechanism: Poisoned = Clean + (Alpha * Edges)
    
    # Normalize images to float for blending
    img_float = img_numpy.astype(np.float32)
    edges_float = edges_rgb.astype(np.float32)
    
    # Add the edges. 
    # Since edges are white (255) on black (0), this lightens the image 
    # where the edges are. You can also subtract to make dark ink.
    poisoned = img_float + (edges_float * alpha)
    
    # Clip to valid range
    poisoned = np.clip(poisoned, 0, 255)

    # --- 5. RETURN TO ORIGINAL FORMAT ---
    if is_tensor:
        poisoned_tensor = torch.from_numpy(poisoned).float()
        poisoned_tensor = poisoned_tensor.permute(2, 0, 1) # Back to C, H, W
        poisoned_tensor = poisoned_tensor / 255.0      # Back to 0.0 - 1.0
        return poisoned_tensor.to(original_device)
        
    return poisoned.astype(np.uint8)






















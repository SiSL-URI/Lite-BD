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


class Autoencoder(nn.Module):
    def __init__(self, channels=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 4), stride=(2, 2), padding=(1, 1), output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1), output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1), output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, (4, 4), stride=(2, 2), padding=(1, 1), output_padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        input_size = x.shape[-2:]  # Save input size (H, W)
        x = self.encoder(x)
        x = self.decoder(x)
        # Resize to match input size exactly
        if x.shape[-2:] != input_size:
            x = torch.nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

    
def add_badnet_trigger(image, trigger_size=6, corner='bottom-right', intensity = 25):
    
    if intensity == 0:
        return image
    
    #trigger_size=4
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pattern_path = 'pattern25.png'
    pattern_image = Image.open(pattern_path).resize((trigger_size, trigger_size))
    pil_image = Image.fromarray(image)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    width, height = pil_image.size
    if corner == 'top-left':
        position = (0, 0)
    elif corner == 'top-right':
        position = (width - trigger_size, 0)
    elif corner == 'bottom-left':
        position = (0, height - trigger_size)
    elif corner == 'bottom-right':
        position = (width - trigger_size, height - trigger_size)
    else:
        raise ValueError("Invalid corner parameter. Choose from 'top-left', 'top-right', 'bottom-left', 'bottom-right'.")

    pil_image.paste(pattern_image, position)
    new_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255

    return new_image


# def add_badnet_trigger(image, trigger_size=4, corner='bottom-right', intensity = 25): #new_badnet
    
#     if intensity == 0:
#         return image
    
#     #trigger_size=4
#     image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#     pattern_path = 'pattern25.png'
#     pattern_image = Image.open(pattern_path).resize((trigger_size, trigger_size))
#     pil_image = Image.fromarray(image)
#     if pil_image.mode != 'RGB':
#         pil_image = pil_image.convert('RGB')

#     width, height = pil_image.size
#     if corner == 'top-left':
#         position = (0, 0)
#     elif corner == 'top-right':
#         position = (width - trigger_size, 0)
#     elif corner == 'bottom-left':
#         position = (0, height - trigger_size)
#     elif corner == 'bottom-right':
#         position = (width - trigger_size, height - trigger_size)
#     else:
#         raise ValueError("Invalid corner parameter. Choose from 'top-left', 'top-right', 'bottom-left', 'bottom-right'.")

#     pil_image.paste(pattern_image, position)
#     new_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255

#     return new_image


# def add_badnet_trigger(image, trigger_size=6, corner='bottom-right', intensity=25, trigger_type='checkboard', opacity=1.0):
#     if intensity == 0:
#         return image

#     assert 0.0 <= opacity <= 1.0, "Opacity must be between 0 and 1."

#     # Convert tensor image to NumPy (HxWxC) and scale to [0, 255]
#     image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#     pil_image = Image.fromarray(image_np).convert('RGB')

#     # Prepare the pattern image
#     if trigger_type == 'checkboard':
#         pattern_path = 'pattern25.png'
#         pattern_image = Image.open(pattern_path).resize((trigger_size, trigger_size)).convert('RGB')
#     elif trigger_type == 'checkboard1':
#         pattern_path = 'pattern_1.png'
#         pattern_image = Image.open(pattern_path).resize((trigger_size, trigger_size)).convert('RGB')
#     elif trigger_type == 'checkboard2':
#         pattern_path = 'pattern_2.png'
#         pattern_image = Image.open(pattern_path).resize((trigger_size, trigger_size)).convert('RGB')
#     elif trigger_type == 'black':
#         pattern_image = Image.fromarray(np.zeros((trigger_size, trigger_size, 3), dtype=np.uint8))
#     elif trigger_type == 'white':
#         pattern_image = Image.fromarray(np.ones((trigger_size, trigger_size, 3), dtype=np.uint8) * 255)
#     else:
#         raise ValueError(f"Invalid trigger_type: {trigger_type}")

#     # Determine position
#     width, height = pil_image.size
#     if corner == 'top-left':
#         position = (0, 0)
#     elif corner == 'top-right':
#         position = (width - trigger_size, 0)
#     elif corner == 'bottom-left':
#         position = (0, height - trigger_size)
#     elif corner == 'bottom-right':
#         position = (width - trigger_size, height - trigger_size)
#     elif corner == 'center' or corner == 'middle':
#         position = ((width - trigger_size) // 2, (height - trigger_size) // 2)
#     else:
#         raise ValueError("Invalid corner parameter.")

#     # Extract region from original image
#     region = pil_image.crop((position[0], position[1], position[0] + trigger_size, position[1] + trigger_size))

#     # Blend the trigger with the original region using opacity
#     blended = Image.blend(region, pattern_image, opacity)

#     # Paste blended patch back into image
#     pil_image.paste(blended, position)

#     # Convert back to PyTorch tensor and normalize to [0, 1]
#     new_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255

#     return new_image

# trigger_pattern_npy = np.load("gtsrb_vgg19_0_255.npy") #np.load("cifar10_preactresnet18_0_255.npy")

# def add_lf_trigger(image):
#     """
#     Adds LF trigger to a PyTorch Tensor.
#     Assumes image is shape (C, H, W).
#     """
    
#     # 1. Convert Numpy Pattern to Tensor
#     # from_numpy creates a tensor sharing memory if possible. 
#     trigger = torch.from_numpy(trigger_pattern_npy).float()
    
#     # 2. Fix Dimensions
#     # Numpy is (H, W, C), but PyTorch Image is (C, H, W).
#     # We must permute the trigger to match the image.
#     trigger = trigger.permute(2, 0, 1)

#     # 3. Handle Device (CPU/GPU)
#     # Ensure trigger is on the same device as the image
#     if isinstance(image, torch.Tensor):
#         trigger = trigger.to(image.device)
        
#         # 4. Handle Range Mismatch (CRITICAL)
#         # transforms.ToTensor() converts images to [0.0, 1.0].
#         # The trigger .npy is usually [0, 255].
#         if image.dtype.is_floating_point and image.max() <= 1.0:
#             # Scale trigger down to 0-1
#             trigger = trigger / 255.0
#             upper_limit = 1.0
#         else:
#             # Assume image is 0-255
#             upper_limit = 255.0

#         # 5. Add and Clamp
#         new_image = image + trigger
#         new_image = torch.clamp(new_image, 0, upper_limit)
        
#         return new_image
    
trigger_pattern_npy = np.load("tiny_preactresnet18_0_255.npy") # np.load("cifar10_preactresnet18_0_255.npy") # np.load("gtsrb_vgg19_0_255.npy")

def add_lf_trigger(image: torch.Tensor) -> torch.Tensor:
    """
    Adds LF trigger to a PyTorch Tensor.
    Assumes image is shape (C, H, W).
    The pattern is resized to match the input image size.
    """
    
    # 1. Convert Numpy Pattern to Tensor
    # from_numpy creates a tensor sharing memory if possible.
    trigger = torch.from_numpy(trigger_pattern_npy).float()
    
    # 2. Fix Dimensions (H, W, C) -> (C, H, W)
    trigger = trigger.permute(2, 0, 1)

    # --- NEW STEP: 3. RESIZE PATTERN TO MATCH INPUT IMAGE (H, W) ---
    
    # Get the target (H, W) dimensions from the image tensor
    target_size = image.shape[1:] # This gives the tuple (H, W), e.g., (32, 32)
    
    # Check if resizing is necessary to avoid unnecessary computation
    if trigger.shape[1] != target_size[0] or trigger.shape[2] != target_size[1]:
        
        # F.interpolate requires input to be (B, C, H, W)
        trigger_resized = trigger.unsqueeze(0) 
        
        # Perform interpolation
        trigger_resized = F.interpolate(
            trigger_resized, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False # Standard for image resizing
        )
        
        # Remove the temporary batch dimension
        trigger = trigger_resized.squeeze(0)
    
    # 4. Handle Device (CPU/GPU)
    # Ensure trigger is on the same device as the image
    if isinstance(image, torch.Tensor):
        trigger = trigger.to(image.device)
        
        # 5. Handle Range Mismatch (CRITICAL)
        if image.dtype.is_floating_point and image.max() <= 1.0:
            # Scale trigger down to 0-1
            trigger = trigger / 255.0
            upper_limit = 1.0
        else:
            # Assume image is 0-255
            upper_limit = 255.0

        # 6. Add and Clamp
        new_image = image + trigger
        # Clamping uses 0 as the lower bound regardless of upper_limit
        new_image = torch.clamp(new_image, 0, upper_limit)
        
        return new_image


#previous badnet add trigger

# def add_badnet_trigger(image, trigger_size=4, corner='bottom-right', intensity = 25, trigger_type = 'checkboard'):
    
#     if intensity == 0:
#         return image
    
#     # Convert tensor image to NumPy (HxWxC) and scale to [0, 255]
#     image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

#     # Load the pattern image
#     if trigger_type == 'checkboard': pattern_path = 'pattern25.png'
#     elif trigger_type == 'checkboard1': pattern_path = 'pattern_1.png'
#     elif trigger_type == 'checkboard2': pattern_path = 'pattern_2.png'
#     pattern_image = Image.open(pattern_path).resize((trigger_size, trigger_size))
    
#     # Convert to PIL for manipulation
#     pil_image = Image.fromarray(image)

#     # Ensure the image is in RGB mode
#     if pil_image.mode != 'RGB':
#         pil_image = pil_image.convert('RGB')

#     # Get dimensions and calculate trigger position
#     width, height = pil_image.size
#     if corner == 'top-left':
#         position = (0, 0)
#     elif corner == 'top-right':
#         position = (width - trigger_size, 0)
#     elif corner == 'bottom-left':
#         position = (0, height - trigger_size)
#     elif corner == 'bottom-right':
#         position = (width - trigger_size, height - trigger_size)
#     elif corner == 'center' or corner == 'middle':
#         position = ((width - trigger_size) // 2, (height - trigger_size) // 2)
#     else:
#         raise ValueError("Invalid corner parameter. Choose from 'top-left', 'top-right', 'bottom-left', 'bottom-right'.")

#     # Add the trigger to the image
#     pil_image.paste(pattern_image, position)

#     # Convert back to PyTorch tensor and normalize to [0, 1]
#     new_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255

#     return new_image


def add_fiba_trigger(img, intensity = 25):
    """
    Apply a frequency-based trigger to an image with different transformations based on intensity.

    Args:
        img (torch.Tensor): Source image tensor of shape (C, H, W).
        intensity (int): Intensity level (between 25 and 225) that controls the strength of the trigger.

    Returns:
        torch.Tensor: Transformed image with trigger applied.
    """
    
    if intensity == 0:
        return img

    # Convert tensor to NumPy array (H, W, C)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Load and preprocess the target image
    target_img = Image.open('pattern25.png').convert("RGB")
    target_img = target_img.resize((img_np.shape[1], img_np.shape[0]))
    target_img = np.asarray(target_img)

    # Scale beta and ratio based on intensity
    beta = 0.05 + (intensity - 25) / 1000  # Maps to range ~ [0.05, 0.2]
    ratio = 0.1 + (intensity - 25) / 250   # Maps to range ~ [0.1, 0.8]

    # Frequency domain manipulation
    fft_trg_cp = np.fft.fft2(target_img, axes=(0, 1))
    amp_target, pha_target = np.abs(fft_trg_cp), np.angle(fft_trg_cp)
    amp_target_shift = np.fft.fftshift(amp_target, axes=(0, 1))

    fft_source_cp = np.fft.fft2(img_np, axes=(0, 1))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    amp_source_shift = np.fft.fftshift(amp_source, axes=(0, 1))

    # Swap amplitude in the central region
    h, w, c = img_np.shape
    b = int(np.floor(min(h, w) * beta))
    c_h, c_w = h // 2, w // 2
    h1, h2 = c_h - b, c_h + b
    w1, w2 = c_w - b, c_w + b

    amp_source_shift[h1:h2, w1:w2, :] = (
        amp_source_shift[h1:h2, w1:w2, :] * (1 - ratio) +
        amp_target_shift[h1:h2, w1:w2, :] * ratio
    )

    # Inverse FFT
    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(0, 1))
    fft_local_ = amp_source_shift * np.exp(1j * pha_source)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(0, 1))
    local_in_trg = np.real(local_in_trg)

    # Clip and convert back to tensor
    local_in_trg = np.clip(local_in_trg, 0, 255).astype(np.uint8)
    transformed_tensor = torch.from_numpy(local_in_trg).permute(2, 0, 1).float() / 255

    return transformed_tensor

def add_blend_trigger(input_image, intensity=25):
    """
    Blend the trigger image onto the input image with a specified intensity.

    Args:
        input_image (torch.Tensor): The base image tensor of shape (C, H, W).
        intensity (int): The blending intensity, ranging from 0 (no blend) to 225 (maximum blend).

    Returns:
        torch.Tensor: The resulting image after blending the trigger image with the input.
    """
    if intensity == 0:
        return input_image
    
    
    # Convert intensity to [0, 1] range
    intensity = intensity / 225
    
    #intensity = 0.05 #new

    # Convert tensor to NumPy array (H, W, C)
    input_np = (input_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Load the trigger image
    trigger_image = cv2.imread('pattern25.png')

    # Ensure the trigger image has the same dimensions as the input image
    if input_np.shape[0] != trigger_image.shape[0] or input_np.shape[1] != trigger_image.shape[1]:
        trigger_image = cv2.resize(trigger_image, (input_np.shape[1], input_np.shape[0]))

    # Blend the images
    blended_image = cv2.addWeighted(input_np, 1 - intensity, trigger_image, intensity, 0)

    # Convert blended image back to tensor
    blended_tensor = torch.from_numpy(blended_image).permute(2, 0, 1).float() / 255

    return blended_tensor

# def add_wanet_trigger(image, intensity=0.25):
    
#     if intensity == 0:
#         return image
    
#     image = (image.permute(1, 2, 0).numpy() * 255)
#     severity = intensity
    
#     # Create a random displacement field
#     displacement = np.random.normal(0, severity, image.shape)
    
#     # Apply the displacement (simple shift)
#     warped_img = np.clip(image+ displacement, 0, 255)
#     new_image = torch.from_numpy(warped_img).permute(2, 0, 1).float() / 255
    
#     return new_image

#modified wanet

def add_wanet_trigger(image, intensity=0.5):
    """
    Apply WaNet-style smooth warping backdoor trigger to a tensor image (C, H, W).
    `intensity` controls the strength of the warp (s parameter in the grid).
    """
    if intensity == 0:
        return image

    # Convert tensor to numpy (H, W, C) in [0, 255] range
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # --- WaNet Trigger Implementation ---
    if not isinstance(img_np, np.ndarray):
        raise TypeError(f"Img should be np.ndarray. Got {type(img_np)}")
    if len(img_np.shape) != 3:
        raise ValueError(f"The shape of img should be HWC. Got {img_np.shape}")

    # Prepare grid
    s = intensity  # warping strength
    k = 32
    grid_rescale = 1
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.interpolate(ins, size=32, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)

    array1d = torch.linspace(-1, 1, steps=32)
    x, y = torch.meshgrid(array1d, array1d, indexing='ij')
    identity_grid = torch.stack((y, x), 2)[None, ...]
    grid = identity_grid + s * noise_grid / 32 * grid_rescale
    grid = torch.clamp(grid, -1, 1)

    # Convert back to torch image
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0
    poison_img = F.grid_sample(img_tensor.unsqueeze(0), grid, align_corners=True).squeeze(0)
    
    return poison_img


# tgtmodel = Autoencoder()
# tgtmodel.to(device)
# tgtmodel.eval()
    
def add_lira_trigger(image, eps=0.05): #add_lira_trigger(image, tgtmodel, device, eps=0.3)
    """
    Add LIRA-style noise trigger to a single image using tgtmodel.
    Args:
        image (torch.Tensor): A single image tensor (C x H x W).
        tgtmodel (nn.Module): The trigger generator model.
        device (torch.device): Device (cuda/cpu).
        eps (float): Strength of the noise.
    Returns:
        torch.Tensor: The poisoned image tensor (C x H x W).
        
    """
    tgtmodel = Autoencoder()
    tgtmodel.to(device)
    checkpoint = torch.load(f'./checkpoint/cifar10_lira_t_3_p_10.0.pth')
    tgtmodel.load_state_dict(checkpoint['tgtmodel'])
    tgtmodel.eval()
    
    # Move image to device and add batch dimension
    image = image.unsqueeze(0).to(device) # (1, C, H, W)

    # Generate noise from trigger model
    noise = tgtmodel(image) * eps
    
    # Ensure same size (safe guard)
    if noise.shape != image.shape:
        noise = torch.nn.functional.interpolate(
            noise, size=image.shape[-2:], mode="bilinear", align_corners=False
        )

    # Add noise and clamp between 0 and 1
    poisoned_image = torch.clamp(image + noise, 0, 1)

    # Remove batch dimension
    poisoned_image = poisoned_image.squeeze(0).cpu()

    return poisoned_image


# def add_trojan_trigger(image):
#     """
#     Applies the TrojanNN trigger to a single image tensor (C x H x W).
    
#     Args:
#         image (torch.Tensor): Normalized image tensor [C x H x W] with values in [0,1].
#         device (str): Device to apply trigger on.
    
#     Returns:
#         torch.Tensor: Image with Trojan trigger applied.
#     """
#     class TrojNN:
#         def __init__(self, shape, device=None):
#             self.device = device
#             self.patch = Image.open('trojnn.jpg')
#             self.patch = torch.Tensor(np.asarray(self.patch) / 255.).permute(2, 0, 1)
#             self.mask = torch.repeat_interleave((self.patch.sum(dim=0, keepdim=True) > 0.3) * 1., 3, dim=0)

#             side_len = shape[1]
#             self.patch = transforms.Resize(side_len)(self.patch)[None, ...].to(self.device)
#             self.mask = transforms.Resize(side_len)(self.mask)[None, ...].to(self.device)
        
#         def inject(self, inputs):
#             out = (1 - self.mask) * inputs + self.mask * self.patch
#             return torch.clamp(out, 0., 1.)

#     model = TrojNN(shape=image.shape, device=device)
#     image = image.to(device).unsqueeze(0)  # Add batch dimension
#     return model.inject(image).squeeze(0).cpu()  # Remove batch dimension and move to CPU

def add_trojan_trigger(image: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    """
    Applies the TrojanNN trigger to a single image tensor (C x H x W).
    
    Args:
        image (torch.Tensor): Normalized image tensor [C x H x W] with values in [0,1].
        device (str): Device to apply trigger on. Defaults to 'cpu'.
    
    Returns:
        torch.Tensor: Image with Trojan trigger applied.
    """
    class TrojNN:
        def __init__(self, shape, device=None):
            self.device = device
            
            # --- Load and Prepare Patch ---
            # NOTE: Ensure 'trojnn.jpg' is in the expected path (or use an absolute path)
            patch_path = 'trojnn.jpg' 
            if not os.path.exists(patch_path):
                 raise FileNotFoundError(f"Trojan patch file not found at: {patch_path}")
                 
            self.patch = Image.open(patch_path).convert('RGB')
            # Convert PIL image to tensor (H x W x C) -> (C x H x W) and normalize
            self.patch = torch.Tensor(np.asarray(self.patch) / 255.).permute(2, 0, 1)
            
            # Create a binary mask based on the patch (where the patch has content)
            self.mask = torch.repeat_interleave((self.patch.sum(dim=0, keepdim=True) > 0.3) * 1., 3, dim=0)

            # --- FIX APPLIED HERE ---
            # Explicitly define target size as (Height, Width) to prevent dimension mismatch errors
            height = shape[1] # H
            width = shape[2]  # W
            target_size = (height, width) 
            
            # Resize patch and mask to the target size (e.g., 32x32)
            self.patch = transforms.Resize(target_size)(self.patch)[None, ...].to(self.device)
            self.mask = transforms.Resize(target_size)(self.mask)[None, ...].to(self.device)
            
        def inject(self, inputs):
            """Applies the trigger using: out = (1 - mask) * inputs + mask * patch"""
            # This operation now has matching (B, C, H, W) shapes for all tensors
            out = (1 - self.mask) * inputs + self.mask * self.patch
            return torch.clamp(out, 0., 1.)

    # Move image to device before calculating shape/initializing model
    image = image.to(device)
    
    # Initialize TrojNN with the image shape
    model = TrojNN(shape=image.shape, device=device)
    
    # Add batch dimension (B, C, H, W) for injection
    image = image.unsqueeze(0) 
    
    # Inject trigger and return the single image tensor back to CPU
    return model.inject(image).squeeze(0).cpu()

import pilgram

def add_filter_trigger(image):
    """
    Applies the Nashville filter from pilgram as a backdoor trigger on CPU.
    
    Args:
        image (torch.Tensor): Normalized image tensor [C x H x W].
    
    Returns:
        torch.Tensor: Image with filter trigger applied.
    """
    class Filter:
        def inject(self, inputs):
            out = inputs.clone()

            # Convert to PIL image after moving to CPU
            out = out[0].cpu().permute((1, 2, 0)).numpy()
            out = np.uint8(out * 255.0)
            out = Image.fromarray(out)

            # Apply Nashville filter
            out = pilgram.nashville(out)

            # Convert back to torch tensor
            out = np.array(out) / 255.0
            out = torch.Tensor(out).permute((2, 0, 1)).unsqueeze(0)
            out = torch.clamp(out, 0., 1.0)
            return out

    f = Filter()
    image = image.cpu().unsqueeze(0)  # [1, C, H, W]
    return f.inject(image).squeeze(0)  # [C, H, W]

def add_badnet_clean_trigger(image, trigger_size=6, intensity=25):
    if intensity == 0:
        return image

    # Convert tensor image to NumPy (HxWxC) and scale to [0, 255]
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Load the pattern image
    pattern_path = 'pattern25.png'
    pattern_image = Image.open(pattern_path).resize((trigger_size, trigger_size))

    # Convert to PIL for manipulation
    pil_image = Image.fromarray(image)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Get dimensions
    width, height = pil_image.size

    # Define four corners
    corners = [
        (0, 0),  # top-left
        (width - trigger_size, 0),  # top-right
        (0, height - trigger_size),  # bottom-left
        (width - trigger_size, height - trigger_size)  # bottom-right
    ]

    # Paste the trigger in all four corners
    for position in corners:
        pil_image.paste(pattern_image, position)

    # Convert back to PyTorch tensor and normalize to [0, 1]
    new_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255

    return new_image

import torchvision

# def add_sig_trigger(image, alpha=0.2):
#     pattern = torch.load('sig.pt')
#     pattern = torchvision.transforms.Resize(image.size(1))(pattern)
#     if pattern.size(1) > 32:
#         alpha = 0.3
#     input = alpha * pattern + (1 - alpha) * image
#     return torch.clamp(input, 0.0, 1.0)

def add_sig_trigger(image: torch.Tensor, alpha: float = 0.2) -> torch.Tensor: # new - 0.05
    """
    Applies a signature trigger pattern to an input image.

    Args:
        image (torch.Tensor): The input image tensor (C x H x W).
        alpha (float): Weight of the pattern in the blended image.

    Returns:
        torch.Tensor: The image with the signature trigger applied.
    """
    pattern = torch.load('sig.pt')
    
    # 1. Determine the exact target dimensions (H and W)
    _, target_h, target_w = image.shape
    target_size = (target_h, target_w)
    
    # 2. Resize the pattern using F.interpolate (Best practice for tensors)
    
    # F.interpolate requires the tensor to be (B, C, H, W).
    # We add a temporary batch dimension (B=1) and remove it afterwards.
    pattern = pattern.unsqueeze(0) 
    
    pattern = F.interpolate(
        pattern, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False # Standard practice for resizing
    ).squeeze(0)
    
    # The pattern is now guaranteed to be (C, H, W) where H and W match the image.
    
    # Check the size (using target_h for clarity) and adjust alpha
    # if target_h > 32:
    #     alpha = 0.3
        
    # 3. Inject the pattern (using 'output' instead of 'input' to avoid shadowing the built-in 'input' function)
    output = alpha * pattern + (1 - alpha) * image
    
    return torch.clamp(output, 0.0, 1.0)


def rnd1(x, decimals=0, out=None):
    """
    Rounds the input array x to the given number of decimals.
    """
    return np.round(x, decimals, out)

def floydDitherspeed(image, squeeze_num):
    """
    Apply Floyd–Steinberg dithering with quantization to reduce the bit depth
    of the image, simulating compression artifacts as a trigger.

    Args:
        image (np.ndarray): Image array with shape (C, H, W) and values in [0, 255].
        squeeze_num (int): Number of quantization levels (e.g., 6).
    
    Returns:
        np.ndarray: Dithered image with the same shape.
    """
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:, y, x]
            temp = np.empty_like(old).astype(np.float64)
            new = rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
            error = old - new
            image[:, y, x] = new
            if x + 1 < w:
                image[:, y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[:, y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:, y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h):
                image[:, y + 1, x - 1] += error * 0.1875
    return image

def add_bppattack_trigger(image, squeeze_num=6):
    """
    Add a BPP-based trigger using dithering compression artifacts.

    Args:
        image (torch.Tensor): A single image tensor of shape (C x H x W).
        squeeze_num (int): Compression level (number of quantization bins).

    Returns:
        torch.Tensor: The poisoned image tensor.
    """
    if squeeze_num <= 1:
        return image  # No transformation

    # Convert to NumPy and scale to [0, 255]
    image_np = image.clone().cpu().numpy() * 255.0
    image_np = image_np.astype(np.float64)

    # Apply dithering-based quantization
    image_np = floydDitherspeed(image_np, squeeze_num)
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    # Convert back to torch tensor and scale to [0, 1]
    poisoned_tensor = torch.from_numpy(image_np).float() / 255.0

    return poisoned_tensor


def add_badnet_trigger_all2all(image, label, trigger_size=6, corner='bottom-right'):
    # Convert tensor image to NumPy (HxWxC) and scale to [0, 255]
    image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Load the class-specific trigger image
    pattern_path = f'pattern_{label}.png'  # pattern_0.jpg to pattern_9.jpg
    pattern_image = Image.open(pattern_path).resize((trigger_size, trigger_size))

    # Convert to PIL
    pil_image = Image.fromarray(image_np)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Place trigger in specified corner
    width, height = pil_image.size
    if corner == 'top-left':
        position = (0, 0)
    elif corner == 'top-right':
        position = (width - trigger_size, 0)
    elif corner == 'bottom-left':
        position = (0, height - trigger_size)
    elif corner == 'bottom-right':
        position = (width - trigger_size, height - trigger_size)
    else:
        raise ValueError("Invalid corner parameter")

    # Add the trigger
    pil_image.paste(pattern_image, position)

    # Convert back to tensor
    new_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255

    return new_image


def add_fiba_trigger_all2all(img, label, intensity=25):
    """
    Apply a frequency-based trigger to an image using a class-specific trigger image.

    Args:
        img (torch.Tensor): Input image of shape (C, H, W).
        label (int): Original class label to determine the trigger.
        intensity (int): Controls the strength of the trigger.

    Returns:
        torch.Tensor: Image with frequency-based trigger applied.
    """
    if intensity == 0:
        return img

    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Load the class-specific trigger image
    pattern_path = f'pattern_{label}.png'
    target_img = Image.open(pattern_path).convert("RGB").resize((img_np.shape[1], img_np.shape[0]))
    target_img = np.asarray(target_img)

    beta = 0.05 + (intensity - 25) / 1000
    ratio = 0.1 + (intensity - 25) / 250

    fft_trg = np.fft.fft2(target_img, axes=(0, 1))
    amp_target, pha_target = np.abs(fft_trg), np.angle(fft_trg)
    amp_target_shift = np.fft.fftshift(amp_target, axes=(0, 1))

    fft_src = np.fft.fft2(img_np, axes=(0, 1))
    amp_source, pha_source = np.abs(fft_src), np.angle(fft_src)
    amp_source_shift = np.fft.fftshift(amp_source, axes=(0, 1))

    h, w, _ = img_np.shape
    b = int(np.floor(min(h, w) * beta))
    c_h, c_w = h // 2, w // 2
    h1, h2, w1, w2 = c_h - b, c_h + b, c_w - b, c_w + b

    amp_source_shift[h1:h2, w1:w2, :] = (
        amp_source_shift[h1:h2, w1:w2, :] * (1 - ratio) +
        amp_target_shift[h1:h2, w1:w2, :] * ratio
    )

    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(0, 1))
    fft_modified = amp_source_shift * np.exp(1j * pha_source)
    result = np.real(np.fft.ifft2(fft_modified, axes=(0, 1)))

    result = np.clip(result, 0, 255).astype(np.uint8)
    return torch.from_numpy(result).permute(2, 0, 1).float() / 255


def add_blend_trigger_all2all(input_image, label, intensity=25):
    """
    Blend a class-specific trigger image onto the input image.

    Args:
        input_image (torch.Tensor): Input image tensor of shape (C, H, W).
        label (int): Original class label to determine the trigger.
        intensity (int): Blending intensity [0-225].

    Returns:
        torch.Tensor: Blended output image.
    """
    if intensity == 0:
        return input_image

    alpha = intensity / 225.0

    input_np = (input_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    pattern_path = f'pattern_{label}.png'
    trigger_image = cv2.imread(pattern_path)
    trigger_image = cv2.resize(trigger_image, (input_np.shape[1], input_np.shape[0]))

    blended = cv2.addWeighted(input_np, 1 - alpha, trigger_image, alpha, 0)
    blended_tensor = torch.from_numpy(blended).permute(2, 0, 1).float() / 255

    return blended_tensor

import cv2
import numpy as np
import torch
import random
from PIL import Image

# ==========================================
# 1. THE PHYSICS ENGINE (Internal Helper)
# ==========================================
def _blend_images_logic(img_t, img_r, max_image_size=560, ghost_rate=0.49, alpha_t=0.4, offset=(0,0), sigma=-1, ghost_alpha=-1):
    """
    The core math from the Refool paper.
    img_t: Clean Target Image (numpy H,W,C uint8)
    img_r: Reflection Image (numpy H,W,C uint8)
    """
    # Convert to float and normalize to 0-1
    t = np.float32(img_t) / 255.
    r = np.float32(img_r) / 255.
    
    # Resize logic to ensure they match
    h, w, _ = t.shape
    # If reflection is smaller/different, resize it to match target
    if r.shape[:2] != t.shape[:2]:
        r = cv2.resize(r, (w, h), interpolation=cv2.INTER_CUBIC)

    # Physics Simulation
    if alpha_t < 0:
        alpha_t = 1. - random.uniform(0.05, 0.45)

    if random.random() < ghost_rate:
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        if offset[0] == 0 and offset[1] == 0:
            offset = (random.randint(3, 8), random.randint(3, 8))
        
        r_1 = np.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)), 'constant', constant_values=0)
        r_2 = np.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)), 'constant', constant_values=0)
        
        if ghost_alpha < 0:
            ghost_alpha = abs(round(random.random()) - random.uniform(0.15, 0.5))

        ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
        
        # Crop back to original size
        ghost_r = ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :]
        
        # Resize if padding calculation slightly shifted dims
        if ghost_r.shape != (h, w, 3):
             ghost_r = cv2.resize(ghost_r, (w, h))

        reflection_mask = ghost_r * (1 - alpha_t)
        blended = reflection_mask + t * alpha_t
        
        # Gamma correction back
        blended = np.clip(np.power(blended, 1 / 2.2), 0, 1)
        
    else:
        # Focal Blur Mode
        if sigma < 0:
            sigma = random.uniform(1, 5)

        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        # Attenuation
        att = 1.08 + np.random.random() / 10.0
        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        
        r_blur = np.clip(r_blur, 0, 1)
        
        # Simple blending for this mode (simplified from original complex kernel code for stability)
        blended = r_blur * (1 - alpha_t) + t * alpha_t
        blended = np.clip(np.power(blended, 1 / 2.2), 0, 1)

    return np.uint8(blended * 255)

# ==========================================
# 2. THE MAIN TRIGGER FUNCTION
# ==========================================

# NOTE: You must provide a path to a reflection image. 
# In the paper, they pick a random image from a folder for every attack.
# For this function, we default to one specific path or you can pass it in.
DEFAULT_REFLECTION_PATH = "./pattern25.png" 

def add_refool_trigger(image, reflection_image=None):
    """
    Adds the Refool trigger. Handles PyTorch Tensors automatically.
    
    Args:
        image: Clean image (Tensor [C,H,W] or Numpy [H,W,C])
        reflection_image: (Optional) Path to reflection image OR numpy array.
                          If None, tries to load DEFAULT_REFLECTION_PATH.
    """
    
    # --- 1. PREPARE INPUT IMAGE ---
    is_tensor = False
    original_device = None
    
    if isinstance(image, torch.Tensor):
        is_tensor = True
        original_device = image.device
        # Tensor (C, H, W) -> Numpy (H, W, C)
        # Clone to cpu, detach gradient, permute
        img_numpy = image.detach().cpu().permute(1, 2, 0).numpy()
        
        # Scale: If tensor is 0.0-1.0, scale to 0-255
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

    # --- 2. PREPARE REFLECTION IMAGE ---
    # Refool NEEDS a second image to act as the reflection.
    
    ref_numpy = None
    
    if reflection_image is None:
        # Load default from disk
        try:
            ref_numpy = cv2.imread(DEFAULT_REFLECTION_PATH)
            if ref_numpy is None: raise FileNotFoundError
            ref_numpy = cv2.cvtColor(ref_numpy, cv2.COLOR_BGR2RGB) # CV2 loads BGR, we need RGB
        except:
            # Fallback: Create random noise if file not found (just so code runs)
            print("Warning: Reflection image not found. Using random noise.")
            ref_numpy = np.random.randint(0, 255, img_numpy.shape, dtype=np.uint8)
            
    elif isinstance(reflection_image, str):
        ref_numpy = cv2.imread(reflection_image)
        ref_numpy = cv2.cvtColor(ref_numpy, cv2.COLOR_BGR2RGB)
    elif isinstance(reflection_image, np.ndarray):
        ref_numpy = reflection_image
        
    # --- 3. APPLY PHYSICS BLENDING ---
    # We use the fixed params from the header description (alpha_t=0.4)
    poisoned_numpy = _blend_images_logic(
        img_numpy, 
        ref_numpy, 
        alpha_t=0.9,
        max_image_size=max(img_numpy.shape[0], img_numpy.shape[1])
    )

    # --- 4. RETURN TO ORIGINAL FORMAT ---
    if is_tensor:
        # Numpy (H, W, C) -> Tensor (C, H, W)
        poisoned_tensor = torch.from_numpy(poisoned_numpy).float()
        poisoned_tensor = poisoned_tensor.permute(2, 0, 1)
        
        # Scale back to 0-1
        poisoned_tensor = poisoned_tensor / 255.0
        
        return poisoned_tensor.to(original_device)
        
    return poisoned_numpy

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































































































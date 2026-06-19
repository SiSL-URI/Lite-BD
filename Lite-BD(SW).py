import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import argparse
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import random
from PIL import Image
from torchvision import datasets, transforms
from collections import defaultdict
from tqdm import tqdm
import copy
import torch.nn.functional as F
import train_test
import resnet
import vgg
import backdoor_triggers
import backdoor_triggers_extended
import gc
import torchvision.transforms.functional as G
import pandas as pd
from sklearn.metrics import accuracy_score
import cv2
from SwinIR.models.network_swinir import SwinIR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('checkpoint', exist_ok=True)

def get_labeled_loader(dataset, target_label, batch_size=32, num_workers=2):
    indices = [i for i, (_, label) in enumerate(dataset) if label == target_label]
    subset = Subset(dataset, indices)
    labeled_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return labeled_loader


# ============================================================================
# STAGE 1: Random Resize & Pad with SwinIR Super-Resolution Recovery
# ============================================================================

def load_swinir_model(model_path='SR_models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
                      device='cuda',
                      lightweight=True,
                      scale=2):
    """
    Load SwinIR model for super-resolution (NEW - from Code 2)
    
    Args:
        model_path: Path to pretrained SwinIR weights
        device: 'cuda' or 'cpu'
        lightweight: True for 60-dim (fast), False for 180-dim (quality)
        scale: Upscaling factor (2, 3, 4, 8)
    
    Returns:
        SwinIR model in eval mode
    """
    print(f"Loading SwinIR model from {model_path}...")
    
    # Model configuration
    if lightweight:
        embed_dim = 60
        upsampler = 'pixelshuffledirect'
        print("Using Lightweight SwinIR (embed_dim=60)")
    else:
        embed_dim = 180
        upsampler = 'pixelshuffle'
        print("Using Medium SwinIR (embed_dim=180)")
    
    # Initialize model
    model = SwinIR(
        upscale=scale,
        img_size=(64, 64),  # Will handle different sizes dynamically
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=embed_dim,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler=upsampler,
        resi_connection='1conv'
    ).to(device)
    
    # Load pretrained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    print(f"✓ SwinIR model loaded successfully!")
    return model


def swinir_upsample_to_size(img_cropped, target_H, target_W, swinir_model, device='cuda'):
    """
    Uses SwinIR to upsample cropped image to target size. (NEW - from Code 2)
    
    Args:
        img_cropped: Tensor (C, H, W) in [0, 1] range (unnormalized)
        target_H, target_W: Target dimensions
        swinir_model: SwinIR model
        device: torch device
    
    Returns:
        Tensor (C, target_H, target_W) in [0, 1] range
    """
    # Ensure input is in [0, 1]
    img_cropped = torch.clamp(img_cropped, 0, 1)
    
    with torch.no_grad():
        # Add batch dimension
        img_batch = img_cropped.unsqueeze(0).to(device)
        
        try:
            # Apply SwinIR (automatic upscaling based on model config)
            sr_img = swinir_model(img_batch)
            sr_img = torch.clamp(sr_img, 0, 1)
            
            # Remove batch dimension
            sr_img = sr_img.squeeze(0)
            
            # Resize to exact target size if needed
            if sr_img.shape[1] != target_H or sr_img.shape[2] != target_W:
                sr_img = F.interpolate(
                    sr_img.unsqueeze(0),
                    size=(target_H, target_W),
                    mode='bicubic',
                    align_corners=False
                ).squeeze(0)
            
        except Exception as e:
            print(f"SwinIR failed: {e}, falling back to bilinear interpolation")
            # Fallback to bilinear if SwinIR fails
            sr_img = F.interpolate(
                img_cropped.unsqueeze(0).to(device),
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
    
    return sr_img


def random_resize_pad_with_sr(batch_tensor, sr_model, min_scale=0.75, return_defended=False):
    """
    Applies Random Resizing and Padding defense with SwinIR Super-Resolution recovery.
    MODIFIED: Now uses SwinIR instead of Real-ESRGAN
    
    Args:
        batch_tensor: Tensor (B, C, H, W) with values in [0, 1] range (unnormalized).
        sr_model: SwinIR model for super-resolution.
        min_scale: Minimum scaling factor.
        return_defended: If True, return defended images; else return recovered images.
        
    Returns:
        Tensor (B, C, H, W): Defended or recovered batch.
    """
    defended_outputs = []
    recovered_outputs = []
    device = batch_tensor.device
    
    # Process each image individually
    for img in batch_tensor:
        C, H, W = img.shape
        
        # --- Step 1: Random Resize & Pad (Defense) ---
        scale = random.uniform(min_scale, min_scale)
        new_H, new_W = max(1, int(H * scale)), max(1, int(W * scale))
        
        # Downsample
        img_down = F.interpolate(
            img.unsqueeze(0), 
            size=(new_H, new_W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Calculate and apply random padding
        pad_H, pad_W = H - new_H, W - new_W
        pad_left = random.randint(0, pad_W)
        pad_right = pad_W - pad_left
        pad_top = random.randint(0, pad_H)
        pad_bottom = pad_H - pad_top
        
        img_defended = F.pad(
            img_down, 
            (pad_left, pad_right, pad_top, pad_bottom), 
            mode='constant', 
            value=0
        )
        defended_outputs.append(img_defended)
        
        # --- Step 2: Recovery with SwinIR Super-Resolution ---
        
        # Crop out the padding
        img_cropped = img_defended[:, pad_top:pad_top+new_H, pad_left:pad_left+new_W]
        
        # SwinIR super-resolution upsample to original size
        img_recovered = swinir_upsample_to_size(img_cropped, H, W, sr_model, device)
        
        recovered_outputs.append(img_recovered)
    
    if return_defended:
        return torch.stack(defended_outputs)
    return torch.stack(recovered_outputs)


# ============================================================================
# STAGE 2: Frequency Band Detection Defense with Recovery Methods
# ============================================================================

def super_resolution_recovery(filtered_img, original_img, alpha=0.3):
    """
    Apply super-resolution by blending high-frequency details from original
    
    Args:
        filtered_img: Image after frequency filtering (C, H, W) - torch.Tensor
        original_img: Original image before filtering (C, H, W) - torch.Tensor
        alpha: Blending weight for high-freq details (0-1)
    """
    # Ensure tensors are on CPU for processing
    filtered_img = filtered_img.cpu()
    original_img = original_img.cpu()
    
    # Extract low-frequency component from original using average pooling
    low_freq_original = F.avg_pool2d(original_img.unsqueeze(0), 
                                     kernel_size=3, stride=1, padding=1).squeeze(0)
    
    # High-frequency details = original - low_frequency
    high_freq_details = original_img - low_freq_original
    
    # Add back some high-frequency details to filtered image
    enhanced = filtered_img + alpha * high_freq_details
    
    # Clamp to valid range [0, 1]
    enhanced = torch.clamp(enhanced, 0, 1)
    
    return enhanced


def bilateral_filter_recovery(filtered_img, sigma_color=0.1, sigma_space=2):
    """
    Apply bilateral filtering to reduce artifacts while preserving edges
    """
    filtered_np = (filtered_img.cpu().numpy() * 255).astype(np.uint8)
    
    result_channels = []
    for c in range(3):
        channel = filtered_np[c]
        # Apply bilateral filter
        filtered_channel = cv2.bilateralFilter(channel, d=5, 
                                               sigmaColor=sigma_color*255, 
                                               sigmaSpace=sigma_space)
        result_channels.append(filtered_channel / 255.0)
    
    result = np.stack(result_channels, axis=0)
    return torch.from_numpy(result).float()


def unsharp_masking(filtered_img, amount=0.5, threshold=0):
    """
    Apply unsharp masking to enhance edges
    """
    filtered_img = filtered_img.cpu()
    
    # Create blurred version
    blurred = F.avg_pool2d(filtered_img.unsqueeze(0), 
                          kernel_size=3, stride=1, padding=1).squeeze(0)
    
    # Create mask (difference between original and blurred)
    mask = filtered_img - blurred
    
    # Apply threshold
    if threshold > 0:
        mask = torch.where(torch.abs(mask) < threshold, 
                          torch.zeros_like(mask), mask)
    
    # Add mask back to image
    sharpened = filtered_img + amount * mask
    sharpened = torch.clamp(sharpened, 0, 1)
    
    return sharpened


def adaptive_frequency_filter_soft(batch_tensor, filter_type='bandstop', 
                                   cutoff_low=None, cutoff_high=None, 
                                   transition_width=0.05):
    """
    Apply frequency filtering with smooth transitions (Gaussian-like rolloff)
    to reduce ringing artifacts
    
    Args:
        batch_tensor: Tensor of shape (B, C, H, W) with values in [0, 1]
        filter_type: 'bandstop' (remove band), 'bandpass' (keep only band)
        cutoff_low: Lower frequency bound (normalized 0-1)
        cutoff_high: Upper frequency bound (normalized 0-1)
        transition_width: Width of transition band (0.01-0.1)
    """
    outputs = []
    
    for img in batch_tensor:
        filtered_channels = []
        
        for c in range(3):
            channel = img[c].cpu().numpy()
            
            # FFT
            f_transform = np.fft.fft2(channel)
            f_shift = np.fft.fftshift(f_transform)
            
            # Create distance map
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2
            y, x = np.ogrid[:rows, :cols]
            distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
            max_distance = np.sqrt(crow**2 + ccol**2)
            distance_normalized = distance / max_distance
            
            # Create SMOOTH mask (not sharp cutoff)
            mask = np.ones((rows, cols), dtype=np.float32)
            
            if filter_type == 'bandstop':
                if cutoff_low is not None and cutoff_high is not None:
                    # Create smooth bandstop mask with gradual transitions
                    in_band = (distance_normalized >= cutoff_low) & \
                             (distance_normalized <= cutoff_high)
                    near_lower = (distance_normalized >= (cutoff_low - transition_width)) & \
                                (distance_normalized < cutoff_low)
                    near_upper = (distance_normalized > cutoff_high) & \
                                (distance_normalized <= (cutoff_high + transition_width))
                    
                    # Hard stop in the middle
                    mask[in_band] = 0.0
                    
                    # Smooth transitions at boundaries
                    if np.any(near_lower):
                        t = (distance_normalized[near_lower] - (cutoff_low - transition_width)) / transition_width
                        mask[near_lower] = 1 - t
                    
                    if np.any(near_upper):
                        t = (distance_normalized[near_upper] - cutoff_high) / transition_width
                        mask[near_upper] = t
            
            elif filter_type == 'bandpass':
                if cutoff_low is not None and cutoff_high is not None:
                    # Inverse of bandstop
                    mask = np.zeros((rows, cols), dtype=np.float32)
                    
                    in_band = (distance_normalized >= cutoff_low) & \
                             (distance_normalized <= cutoff_high)
                    near_lower = (distance_normalized >= (cutoff_low - transition_width)) & \
                                (distance_normalized < cutoff_low)
                    near_upper = (distance_normalized > cutoff_high) & \
                                (distance_normalized <= (cutoff_high + transition_width))
                    
                    mask[in_band] = 1.0
                    
                    if np.any(near_lower):
                        t = (distance_normalized[near_lower] - (cutoff_low - transition_width)) / transition_width
                        mask[near_lower] = t
                    
                    if np.any(near_upper):
                        t = (distance_normalized[near_upper] - cutoff_high) / transition_width
                        mask[near_upper] = 1 - t
            
            # Apply smooth mask
            f_shift_filtered = f_shift * mask
            
            # Inverse FFT
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)
            img_back = np.clip(img_back, 0, 1)
            
            filtered_channels.append(img_back)
        
        filtered_img = np.stack(filtered_channels, axis=0)
        filtered_tensor = torch.from_numpy(filtered_img).float()
        outputs.append(filtered_tensor)
    
    return torch.stack(outputs)


def frequency_filter(batch_tensor, filter_type='bandstop', cutoff_low=None, cutoff_high=None):
    """
    Apply frequency domain filtering (sharp cutoff - original version)
    Args:
        batch_tensor: Tensor of shape (B, C, H, W) with values in [0, 1]
        filter_type: 'bandstop' (remove band), 'bandpass' (keep only band)
        cutoff_low: Lower frequency bound (normalized 0-1)
        cutoff_high: Upper frequency bound (normalized 0-1)
    """
    outputs = []
    
    for img in batch_tensor:
        filtered_channels = []
        
        for c in range(3):  # RGB channels
            channel = img[c].cpu().numpy()
            
            # 1. Apply 2D FFT
            f_transform = np.fft.fft2(channel)
            f_shift = np.fft.fftshift(f_transform)
            
            # 2. Create frequency filter mask
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2
            
            # Calculate distance from center
            y, x = np.ogrid[:rows, :cols]
            distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
            
            # Normalize distance to [0, 1]
            max_distance = np.sqrt(crow**2 + ccol**2)
            distance_normalized = distance / max_distance
            
            # Create mask based on filter type
            mask = np.ones((rows, cols), dtype=np.float32)
            
            if filter_type == 'bandstop':
                # Remove frequencies in [cutoff_low, cutoff_high]
                if cutoff_low is not None and cutoff_high is not None:
                    mask[(distance_normalized >= cutoff_low) & 
                         (distance_normalized <= cutoff_high)] = 0
                    
            elif filter_type == 'bandpass':
                # Keep only frequencies in [cutoff_low, cutoff_high]
                if cutoff_low is not None and cutoff_high is not None:
                    mask[(distance_normalized < cutoff_low) | 
                         (distance_normalized > cutoff_high)] = 0
            
            # 3. Apply mask in frequency domain
            f_shift_filtered = f_shift * mask
            
            # 4. Inverse FFT
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)
            img_back = np.clip(img_back, 0, 1)
            
            filtered_channels.append(img_back)
        
        filtered_img = np.stack(filtered_channels, axis=0)
        filtered_tensor = torch.from_numpy(filtered_img).float()
        outputs.append(filtered_tensor)
    
    return torch.stack(outputs)


def find_trigger_frequency_band(img_tensor, net, target_label, clean_label, 
                                mean, std, device='cuda', num_bands=20,
                                use_smooth_filter=True,
                                transition_width=0.05,
                                recovery_method='super_resolution',
                                recovery_strength=0.3,
                                verbose=False):
    """
    Systematically test each frequency band to find which one contains the trigger.
    Now includes smooth filtering and spatial resolution recovery.
    
    Args:
        img_tensor: Image tensor (C, H, W), normalized
        net: Model
        target_label: Backdoor target class
        clean_label: Original clean class
        mean, std: Normalization parameters
        device: cuda/cpu
        num_bands: Number of frequency bands to test
        use_smooth_filter: Use smooth transitions instead of sharp cutoffs
        transition_width: Width of smooth transition (0.01-0.1)
        recovery_method: 'super_resolution', 'bilateral', 'unsharp', 'none'
        recovery_strength: Strength of recovery (0-1)
        verbose: Print detailed info
    
    Returns:
        suspicious_bands: List of band info dicts
        filtered_image: Image with most suspicious band removed + recovery applied
    """
    net.eval()
    
    # Unnormalize
    img_unnorm = img_tensor * std + mean
    img_unnorm = torch.clamp(img_unnorm, 0, 1)
    
    # Define frequency bands
    band_edges = np.linspace(0, 1, num_bands + 1)
    
    # Store results
    suspicious_bands = []
    
    # Track prediction frequency when each band is removed
    prediction_counts = defaultdict(int)
    
    if verbose:
        print(f"\nTesting {num_bands} frequency bands...")
        print(f"Smooth filter: {use_smooth_filter}, Recovery: {recovery_method}")
        print(f"Target label: {target_label}, Clean label: {clean_label}")
        print("-" * 80)
    
    # Choose filter function
    filter_func = adaptive_frequency_filter_soft if use_smooth_filter else frequency_filter
    
    with torch.no_grad():
        # Test each band
        for i in range(num_bands):
            lower = band_edges[i]
            upper = band_edges[i + 1]
            
            # Remove this band
            if use_smooth_filter:
                filtered = filter_func(
                    img_unnorm.unsqueeze(0),
                    filter_type='bandstop',
                    cutoff_low=lower,
                    cutoff_high=upper,
                    transition_width=transition_width
                )
            else:
                filtered = filter_func(
                    img_unnorm.unsqueeze(0),
                    filter_type='bandstop',
                    cutoff_low=lower,
                    cutoff_high=upper
                )
            
            # Move to device and renormalize
            filtered = filtered.to(device)
            filtered_norm = (filtered - mean) / std
            
            # Get prediction
            output = net(filtered_norm)
            _, pred = output.max(1)
            pred_label = pred.item()
            
            # Check if label flipped from target
            label_flipped = (pred_label != target_label)
            
            if verbose:
                status = "✓ FLIP!" if label_flipped else ""
                print(f"Band [{lower:.3f}, {upper:.3f}]: "
                      f"pred={pred_label} {status}")
            
            if label_flipped:
                # Count predictions
                prediction_counts[pred_label] += 1
                
                suspicious_bands.append({
                    'lower': lower,
                    'upper': upper,
                    'pred_label': pred_label,
                    'band_center': (lower + upper) / 2,
                    'bandwidth': upper - lower,
                    'band_idx': i
                })
    
    if verbose:
        print("-" * 80)
        if suspicious_bands:
            print(f"\n✓ Found {len(suspicious_bands)} suspicious band(s):")
            for band in suspicious_bands:
                print(f"  [{band['lower']:.3f}, {band['upper']:.3f}] → pred={band['pred_label']}")
            
            # Show prediction frequency
            print("\nPrediction frequency when bands are removed:")
            for pred_label, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  Label {pred_label}: {count} bands")
        else:
            print("\n✗ No frequency band found that flips the label")
    
    # Select the best band based on most frequent prediction
    if suspicious_bands:
        # Find the most frequent prediction (excluding target_label)
        if prediction_counts:
            most_common_pred = max(prediction_counts.items(), key=lambda x: x[1])[0]
            
            if verbose:
                print(f"\nMost common flipped prediction: {most_common_pred} " +
                      f"({prediction_counts[most_common_pred]} bands)")
            
            # Filter suspicious bands to only include those with the most common prediction
            best_candidates = [b for b in suspicious_bands if b['pred_label'] == most_common_pred]
            
            if best_candidates:
                best_band = best_candidates[0]
                
                if verbose:
                    print(f"\nSelected band: [{best_band['lower']:.3f}, {best_band['upper']:.3f}] " +
                          f"(pred={best_band['pred_label']})")
            else:
                # Fallback to first suspicious band
                best_band = suspicious_bands[0]
        else:
            # Fallback to first suspicious band
            best_band = suspicious_bands[0]
        
        # Apply bandstop filter to remove the selected band
        if use_smooth_filter:
            filtered_img = filter_func(
                img_unnorm.unsqueeze(0),
                filter_type='bandstop',
                cutoff_low=best_band['lower'],
                cutoff_high=best_band['upper'],
                transition_width=transition_width
            ).squeeze(0)
        else:
            filtered_img = filter_func(
                img_unnorm.unsqueeze(0),
                filter_type='bandstop',
                cutoff_low=best_band['lower'],
                cutoff_high=best_band['upper']
            ).squeeze(0)
        
        # Apply spatial resolution recovery
        if recovery_method == 'super_resolution':
            filtered_img = super_resolution_recovery(
                filtered_img, img_unnorm, alpha=recovery_strength
            )
            if verbose:
                print(f"Applied super-resolution recovery (alpha={recovery_strength})")
        
        elif recovery_method == 'bilateral':
            filtered_img = bilateral_filter_recovery(
                filtered_img, sigma_color=0.1, sigma_space=2
            )
            if verbose:
                print(f"Applied bilateral filter recovery")
        
        elif recovery_method == 'unsharp':
            filtered_img = unsharp_masking(
                filtered_img, amount=recovery_strength, threshold=0
            )
            if verbose:
                print(f"Applied unsharp masking (amount={recovery_strength})")
        
        elif recovery_method == 'none':
            if verbose:
                print(f"No recovery applied")
        
        if verbose:
            print(f"\nFinal filter applied: bandstop [{best_band['lower']:.3f}, {best_band['upper']:.3f}]")
    else:
        # No suspicious band found, return original
        filtered_img = img_unnorm
    
    return suspicious_bands, filtered_img


# ============================================================================
# COMBINED TWO-STAGE DEFENSE PIPELINE
# ============================================================================

def two_stage_defense_pipeline(img_tensor, net, target_label, clean_label,
                               mean, std, sr_model, device='cuda',
                               # Stage 1 params
                               min_scale=0.75,
                               # Stage 2 params
                               num_bands=20,
                               use_smooth_filter=True,
                               transition_width=0.05,
                               recovery_method='super_resolution',
                               recovery_strength=0.3,
                               verbose=False):
    """
    Two-stage defense pipeline with SwinIR:
    MODIFIED: Now uses SwinIR (sr_model parameter) instead of Real-ESRGAN
    1. Stage 1: Random Resize & Pad with SwinIR SR Recovery
    2. Stage 2: Frequency Band Detection (only if stage 1 doesn't flip label)
    3. If both fail: Return original image
    
    Args:
        img_tensor: Normalized image tensor (C, H, W)
        net: Model
        target_label: Backdoor target class
        clean_label: Original clean class
        mean, std: Normalization parameters
        sr_model: SwinIR model for super-resolution
        device: cuda/cpu
        min_scale: Minimum scaling factor for stage 1
        num_bands: Number of frequency bands for stage 2
        use_smooth_filter: Use smooth filtering in stage 2
        transition_width: Transition width for stage 2
        recovery_method: Recovery method for stage 2
        recovery_strength: Recovery strength for stage 2
        verbose: Print detailed info
        
    Returns:
        recovered_img: Final recovered image (unnormalized, [0,1])
        stage_used: 0 (original), 1 (stage 1), or 2 (stage 2)
        pred_label: Final prediction label
    """
    net.eval()
    
    if verbose:
        print("\n" + "="*80)
        print("TWO-STAGE DEFENSE PIPELINE (with SwinIR)")
        print("="*80)

    output_stage1 = net(img_tensor.unsqueeze(0))
    _, pred_before = output_stage1.max(1)
    pred_before = pred_before.item()
    
    #print(pred_before)
    
    # Unnormalize input
    img_unnorm = img_tensor * std + mean
    img_unnorm = torch.clamp(img_unnorm, 0, 1)
    
    # ========================================================================
    # STAGE 1: Random Resize & Pad with SwinIR SR Recovery
    # ========================================================================
    
    if verbose:
        print("\n[STAGE 1] Random Resize & Pad with SwinIR SR Recovery")
        print("-" * 80)
    
    with torch.no_grad():
        # Apply stage 1 defense
        stage1_recovered = random_resize_pad_with_sr(
            img_unnorm.unsqueeze(0), 
            sr_model,  # SwinIR model
            min_scale=min_scale, 
            return_defended=False
        ).squeeze(0)
        
        # Normalize and get prediction
        stage1_norm = (stage1_recovered.to(device) - mean) / std
        output_stage1 = net(stage1_norm.unsqueeze(0))
        _, pred_stage1 = output_stage1.max(1)
        pred_stage1 = pred_stage1.item()
        
        if verbose:
            print(f"Stage 1 prediction: {pred_stage1}")
            print(f"Target label: {target_label}")
            print(f"Clean label: {clean_label}")
        
        # Check if label changed from target
        if pred_stage1 != pred_before:  # target_label:
            if verbose:
                print(f"✓ Stage 1 SUCCESS: Label flipped from {target_label} to {pred_stage1}")
                print(f"Using Stage 1 output as final result")
                print("="*80)
            return stage1_recovered, 1, pred_stage1
        else:
            if verbose:
                print(f"✗ Stage 1 FAILED: Label still {target_label}")
                print(f"Proceeding to Stage 2...")
    
    # ========================================================================
    # STAGE 2: Frequency Band Detection Defense
    # ========================================================================
    
    if verbose:
        print("\n[STAGE 2] Frequency Band Detection Defense")
        print("-" * 80)
    
    # Use the original image as input to stage 2
    stage1_norm_tensor = (img_unnorm.to(device) - mean) / std
    
    suspicious_bands, stage2_recovered = find_trigger_frequency_band(
        stage1_norm_tensor,
        net,
        target_label,
        clean_label,
        mean,
        std,
        device=device,
        num_bands=num_bands,
        use_smooth_filter=use_smooth_filter,
        transition_width=transition_width,
        recovery_method=recovery_method,
        recovery_strength=recovery_strength,
        verbose=verbose
    )
    
    # Get final prediction
    with torch.no_grad():
        stage2_norm = (stage2_recovered.to(device) - mean) / std
        output_stage2 = net(stage2_norm.unsqueeze(0))
        _, pred_stage2 = output_stage2.max(1)
        pred_stage2 = pred_stage2.item()
    
    if verbose:
        print(f"\nStage 2 prediction: {pred_stage2}")
    
    # Check if stage 2 succeeded
    if pred_stage2 != pred_before:  # target_label:
        if verbose:
            print(f"✓ Stage 2 SUCCESS: Label flipped from {target_label} to {pred_stage2}")
            print(f"Using Stage 2 output as final result")
            print("="*80)
        return stage2_recovered, 2, pred_stage2
    else:
        if verbose:
            print(f"✗ Stage 2 FAILED: Label still {target_label}")
            print(f"Both stages failed - returning ORIGINAL image")
            print("="*80)
        
        # Both stages failed - return original image
        return img_unnorm, 0, pred_stage2


def evaluate_two_stage_defense(args, net, poisoned_testloader, clean_labels,
                               target_label, sr_model, device='cuda', N=100,
                               # Stage 1 params
                               min_scale=0.75,
                               # Stage 2 params
                               num_bands=20,
                               use_smooth_filter=True,
                               transition_width=0.05,
                               recovery_method='super_resolution',
                               recovery_strength=0.3,
                               verbose_per_image=False):
    """
    Evaluate the two-stage defense pipeline with SwinIR
    MODIFIED: Now uses SwinIR (sr_model parameter)
    """
    # Get mean and std based on dataset
    if args.dataset == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(3, 1, 1)
        std = torch.tensor([0.247, 0.243, 0.261], device=device).view(3, 1, 1)
    elif args.dataset == 'tiny-imagenet':
        mean = torch.tensor([0.4802, 0.4481, 0.3975], device=device).view(3, 1, 1)
        std = torch.tensor([0.2302, 0.2265, 0.2262], device=device).view(3, 1, 1)
    elif args.dataset == 'gtsrb':
        mean = torch.tensor([0.3403, 0.3121, 0.3214], device=device).view(3, 1, 1)
        std = torch.tensor([0.2724, 0.2608, 0.2669], device=device).view(3, 1, 1)
    elif args.dataset == 'imagenet12':
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    elif args.dataset == 'fer':
        mean = torch.tensor([0.5071], device=device)
        std = torch.tensor([0.2554], device=device)
    elif args.dataset == 'fashion-mnist':
        mean = torch.tensor([0.2860], device=device)
        std = torch.tensor([0.3530], device=device)
    elif args.dataset == 'color-mnist' or 'mnist':
        mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
        
    
    clean_correct = 0
    asr_hit = 0
    total = 0
    
    # Track which stage was used
    stage0_count = 0  # Original (both stages failed)
    stage1_count = 0  # Stage 1 succeeded
    stage2_count = 0  # Stage 2 succeeded
    
    net.eval()
    
    idx = 0
    progress_bar = tqdm(
        total=N,
        desc=f"Two-Stage Defense (SwinIR) [{args.dataset}][{args.atk}]",
        unit="img"
    )
    
    with torch.no_grad():
        for images, _ in poisoned_testloader:
            if total >= N:
                break
            
            images = images.to(device)
            
            for i in range(len(images)):
                if total >= N:
                    break
                
                img_tensor = images[i]
                clean_label = clean_labels[idx]
                if clean_label == target_label:  # Skip samples with same clean and target label
                    idx += 1
                    continue
                
                # Apply two-stage defense with SwinIR
                recovered_img, stage_used, pred = two_stage_defense_pipeline(
                    img_tensor,
                    net,
                    target_label,
                    clean_label,
                    mean,
                    std,
                    sr_model,  # SwinIR model
                    device=device,
                    min_scale=min_scale,
                    num_bands=num_bands,
                    use_smooth_filter=use_smooth_filter,
                    transition_width=transition_width,
                    recovery_method=recovery_method,
                    recovery_strength=recovery_strength,
                    verbose=verbose_per_image
                )
                
                # Track stage usage
                if stage_used == 0:
                    stage0_count += 1
                elif stage_used == 1:
                    stage1_count += 1
                else:
                    stage2_count += 1
                
                # Evaluate
                if pred == clean_label:
                    clean_correct += 1
                if pred == target_label:
                    asr_hit += 1
                
                idx += 1
                total += 1
                progress_bar.update(1)
    
    progress_bar.close()
    
    # Calculate metrics
    if total > 0:
        clean_acc = clean_correct / total
        asr = asr_hit / total
    else:
        clean_acc = 0.0
        asr = 0.0
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"Two-Stage Defense Results (with SwinIR)")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}, Attack: {args.atk}")
    print(f"Images evaluated: {total}")
    print(f"\nStage Usage:")
    print(f"  Stage 0 (Original - Both Failed): {stage0_count}/{total} ({stage0_count/total*100:.1f}%)")
    print(f"  Stage 1 (Resize+Pad+SwinIR): {stage1_count}/{total} ({stage1_count/total*100:.1f}%)")
    print(f"  Stage 2 (Freq Band): {stage2_count}/{total} ({stage2_count/total*100:.1f}%)")
    print(f"\nPerformance:")
    print(f"  Defense PA (Preserved Accuracy): {clean_acc:.4f}")
    print(f"  Defense ASR (Attack Success Rate): {asr:.4f}")
    print(f"{'='*80}\n")
    
    return clean_acc, asr, stage0_count, stage1_count, stage2_count

import matplotlib.pyplot as plt

def visualize_defense_samples(args, net, poisoned_testloader, clean_labels,
                              target_label, sr_model, device='cuda',
                              num_samples=5,
                              # Stage 1 params
                              min_scale=0.75,
                              # Stage 2 params
                              num_bands=20,
                              use_smooth_filter=True,
                              transition_width=0.05,
                              recovery_method='super_resolution',
                              recovery_strength=0.3):
    """
    Visualize samples before and after two-stage defense with SwinIR
    MODIFIED: Now uses SwinIR (sr_model parameter)
    
    Args:
        num_samples: Number of samples to visualize
        Other args: Same as two_stage_defense_pipeline
    """
    # Get mean and std based on dataset
    if args.dataset == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(3, 1, 1)
        std = torch.tensor([0.247, 0.243, 0.261], device=device).view(3, 1, 1)
    elif args.dataset == 'tiny-imagenet':
        mean = torch.tensor([0.4802, 0.4481, 0.3975], device=device).view(3, 1, 1)
        std = torch.tensor([0.2302, 0.2265, 0.2262], device=device).view(3, 1, 1)
    elif args.dataset == 'gtsrb':
        mean = torch.tensor([0.3403, 0.3121, 0.3214], device=device).view(3, 1, 1)
        std = torch.tensor([0.2724, 0.2608, 0.2669], device=device).view(3, 1, 1)
    elif args.dataset == 'imagenet12':
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    elif args.dataset == 'fer':
        mean = torch.tensor([0.5071], device=device).view(1, 1, 1)
        std = torch.tensor([0.2554], device=device).view(1, 1, 1)
    elif args.dataset == 'fashion-mnist':
        mean = torch.tensor([0.2860], device=device)
        std = torch.tensor([0.3530], device=device)
    elif args.dataset == 'color-mnist' or 'mnist':
        mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
    
    net.eval()
    
    # Collect samples
    samples_before = []
    samples_after = []
    predictions_before = []
    predictions_after = []
    clean_label_list = []
    stage_used_list = []
    
    idx = 0
    with torch.no_grad():
        for images, _ in poisoned_testloader:
            if len(samples_before) >= num_samples:
                break
            
            images = images.to(device)
            
            for i in range(len(images)):
                if len(samples_before) >= num_samples:
                    break
                
                img_tensor = images[i]
                clean_label = clean_labels[idx]
                
                # Get prediction before defense
                output_before = net(img_tensor.unsqueeze(0))
                _, pred_before = output_before.max(1)
                pred_before = pred_before.item()
                
                # Apply two-stage defense with SwinIR
                recovered_img, stage_used, pred_after = two_stage_defense_pipeline(
                    img_tensor,
                    net,
                    target_label,
                    clean_label,
                    mean,
                    std,
                    sr_model,  # SwinIR model
                    device=device,
                    min_scale=min_scale,
                    num_bands=num_bands,
                    use_smooth_filter=use_smooth_filter,
                    transition_width=transition_width,
                    recovery_method=recovery_method,
                    recovery_strength=recovery_strength,
                    verbose=False
                )
                
                # Unnormalize before image for visualization
                img_unnorm = img_tensor * std + mean
                img_unnorm = torch.clamp(img_unnorm, 0, 1)
                
                # Store samples
                samples_before.append(img_unnorm.cpu())
                samples_after.append(recovered_img.cpu())
                predictions_before.append(pred_before)
                predictions_after.append(pred_after)
                clean_label_list.append(clean_label)
                stage_used_list.append(stage_used)
                
                idx += 1
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Before defense
        img_before = samples_before[i].permute(1, 2, 0).numpy()
        
        # Handle grayscale (FER)
        if img_before.shape[2] == 1:
            img_before = img_before.squeeze(-1)
            cmap = 'gray'
        else:
            cmap = None
        
        axes[i, 0].imshow(img_before, cmap=cmap)
        axes[i, 0].set_title(
            f'Before Defense\n'
            f'Pred: {predictions_before[i]} (Target: {target_label}, Clean: {clean_label_list[i]})',
            fontsize=10
        )
        axes[i, 0].axis('off')
        
        # After defense
        img_after = samples_after[i].permute(1, 2, 0).numpy()
        
        # Handle grayscale (FER)
        if img_after.shape[2] == 1:
            img_after = img_after.squeeze(-1)
        
        axes[i, 1].imshow(img_after, cmap=cmap)
        
        # Determine success/failure
        success_str = "✓ SUCCESS" if predictions_after[i] != target_label else "✗ FAILED"
        stage_str = f"Stage {stage_used_list[i]}"
        
        axes[i, 1].set_title(
            f'After Defense ({stage_str} - SwinIR)\n'
            f'Pred: {predictions_after[i]} {success_str}',
            fontsize=10
        )
        axes[i, 1].axis('off')
    
    plt.suptitle(
        f'Two-Stage Defense Visualization (with SwinIR)\n'
        f'Dataset: {args.dataset.upper()}, Attack: {args.atk.upper()}',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EVALUATION
# ============================================================================

parser = argparse.ArgumentParser(description='Two-Stage Defense Evaluation with SwinIR.')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--atk', type=str, default='badnet', help='Backdoor attack')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
parser.add_argument('--data_dir', type=str, default='cifar10/', help='Data Path Directory')
parser.add_argument('--t_b', type=int, default=3, help='Target Label')
parser.add_argument('--p', type=float, default=1.0, help='Poison Ratio')
parser.add_argument('--trigger_size', type=int, default=6, help='Trigger Size')
parser.add_argument('--opacity', type=float, default=1, help='Opacity of Trigger')
parser.add_argument('--corner', type=str, default='bottom-right', help='Trigger Position')
parser.add_argument('--trigger_type', type=str, default='checkboard', help='Trigger Type')
parser.add_argument('--n_eval', type=int, default=100, help='Number of images to evaluate')

# SwinIR params (NEW - from Code 2)
parser.add_argument('--swinir_path', type=str, 
                    default='SR_models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
                    help='Path to SwinIR pretrained weights')
parser.add_argument('--swinir_lightweight', action='store_true', default=True,
                    help='Use lightweight SwinIR (60-dim) instead of medium (180-dim)')
parser.add_argument('--swinir_scale', type=int, default=2,
                    help='SwinIR upscaling factor (2, 3, 4, 8)')

# Stage 1 params
parser.add_argument('--min_scale', type=float, default=0.5, help='Minimum scale factor for stage 1')

# Stage 2 params
parser.add_argument('--num_bands', type=int, default=50, help='Number of frequency bands for stage 2')
parser.add_argument('--recovery_method', type=str, default='unsharp',
                    choices=['super_resolution', 'bilateral', 'unsharp', 'none'],
                    help='Spatial resolution recovery method for stage 2')
parser.add_argument('--recovery_strength', type=float, default=0.5,
                    help='Recovery strength for stage 2 (0-1)')
parser.add_argument('--use_smooth_filter', action='store_true', default=True,
                    help='Use smooth frequency transitions in stage 2')
parser.add_argument('--transition_width', type=float, default=0.08,
                    help='Smooth transition width for stage 2 (0.01-0.1)')

parser.add_argument('--verbose_per_image', action='store_true', help='Print detailed info for each image')

args = parser.parse_args()

for action in parser._actions:
    if action.dest != 'help':
        value = getattr(args, action.dest)
        print(f"{action.help}: {value}")

from torchvision.transforms import ToPILImage

class PoisonedDataset(Dataset):
    def __init__(self, args, imagefolder_dataset, trigger_function, target_label):
        self.args = args
        self.dataset = imagefolder_dataset
        self.trigger_function = trigger_function
        self.target_label = target_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, _ = self.dataset.samples[idx]
        img = Image.open(path).convert('RGB')

        img_tensor = transforms.ToTensor()(img)
        
        if self.args.atk == 'lira': poisoned_tensor = self.trigger_function(img_tensor)
        
        else: poisoned_tensor = self.trigger_function(img_tensor)

        poisoned_pil = ToPILImage()(poisoned_tensor)

        if self.dataset.transform:
            poisoned_img = self.dataset.transform(poisoned_pil)
        else:
            poisoned_img = poisoned_tensor

        return poisoned_img, self.target_label


# Main evaluation loop
attacks = ['badnet', 'blend', 'wanet', 'sig', 'cl', 'bppattack', 'trojan', 'lf',  'poison-ink', 'lira']
datasets = ['cifar10', 'gtsrb', 'fashion-mnist']

# Store results
results = []

# MODIFIED: Load SwinIR model ONCE at the beginning (from Code 2)
print(f"\n{'='*100}")
print(f"LOADING SWINIR MODEL")
print(f"{'='*100}")
print(f"Model path: {args.swinir_path}")
print(f"Lightweight: {args.swinir_lightweight}")
print(f"Scale: {args.swinir_scale}x")

swinir_model = load_swinir_model(
    model_path=args.swinir_path,
    device=device,
    lightweight=args.swinir_lightweight,
    scale=args.swinir_scale
)

print(f"{'='*100}\n")

for dataset_name in datasets:
    args.dataset = dataset_name
    
    # Configure dataset-specific parameters
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
        data_dir = 'cifar10/'
        net = resnet.ResNet18(num_class=10)
    
        
    
    elif args.dataset == 'gtsrb':
        gtsrb_mean = [0.3403, 0.3121, 0.3214]
        gtsrb_std = [0.2724, 0.2608, 0.2669]
        
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=gtsrb_mean, std=gtsrb_std),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=gtsrb_mean, std=gtsrb_std),
        ])
        
        data_dir = 'GTSRB/'
        net = vgg.VGG('VGG11')
        
    elif args.dataset == 'fashion-mnist':
        
        # Fashion-MNIST statistics (grayscale)
        fashion_mnist_mean = [0.2860]
        fashion_mnist_std  = [0.3530]

        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),          # for consistency with CIFAR-style models
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=fashion_mnist_mean,
                                 std=fashion_mnist_std),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=fashion_mnist_mean,
                                 std=fashion_mnist_std),
        ])

        data_dir = 'fashion_mnist/'
        
        num_classes = 10
        net = resnet.ResNet18(num_class=10) 

    # Load datasets
    train_dataset = ImageFolder(root=data_dir + '/train', transform=transform_train)
    test_dataset = ImageFolder(root=data_dir + '/test', transform=transform_test)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                             shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                            shuffle=False, num_workers=0)
    
    net = net.to(device)
    
    for atk in attacks:
        args.atk = atk
        args.p = 10.0
        
        print(f"\n{'='*80}")
        print(f"Evaluating: Dataset={args.dataset}, Attack={args.atk}")
        print(f"{'='*80}\n")
        
        # Set trigger function
        if args.atk == 'badnet':
            trig_fun = backdoor_triggers.add_badnet_trigger
        elif args.atk == 'lf':
            trig_fun = backdoor_triggers.add_lf_trigger
        elif args.atk == 'wanet':
            trig_fun = backdoor_triggers.add_wanet_trigger
        elif args.atk == 'blend':
            trig_fun = backdoor_triggers.add_blend_trigger
        elif args.atk == 'fiba':
            trig_fun = backdoor_triggers.add_fiba_trigger
        elif args.atk == 'bppattack':
            trig_fun = backdoor_triggers.add_bppattack_trigger
        elif args.atk == 'sig':
            trig_fun = backdoor_triggers.add_sig_trigger
        elif args.atk == 'cl':
            trig_fun = backdoor_triggers.add_badnet_clean_trigger
        elif args.atk == 'trojan':
            trig_fun = backdoor_triggers.add_trojan_trigger
        elif args.atk == 'filter':
            trig_fun = backdoor_triggers.add_filter_trigger
        elif args.atk == 'lira':
            trig_fun = backdoor_triggers.add_lira_trigger
        elif args.atk == 'refool':
            trig_fun = backdoor_triggers.add_refool_trigger
        elif args.atk == 'poison-ink':
            trig_fun = backdoor_triggers_extended.add_poison_ink_trigger
        
        # Create poisoned test loader
        base_test_dataset = ImageFolder(root=data_dir + '/test', transform=None)
        poisoned_dataset = PoisonedDataset(
            args,
            imagefolder_dataset=base_test_dataset,
            trigger_function=trig_fun,
            target_label=args.t_b
        )
        poisoned_dataset.dataset.transform = transform_test
        poisoned_testloader = torch.utils.data.DataLoader(
            poisoned_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        #poisoned_testloader = testloader  # for ca
        
        # Load backdoored model
        checkpoint_path = f'./checkpoint/{args.dataset}_{args.atk}_t_{args.t_b}_p_{args.p}.pth'
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Skipping this combination...\n")
            continue
            
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        
        # Get clean labels
        clean_targets = [label for _, label in testloader.dataset.samples]
        
        # Visualize defense samples (MODIFIED: Now uses SwinIR)
        visualize_defense_samples(
                args,
                net,
                poisoned_testloader,
                clean_targets,
                target_label=args.t_b,
                sr_model=swinir_model,  # SwinIR model
                device=device,
                num_samples=5,
                min_scale=args.min_scale,
                num_bands=args.num_bands,
                use_smooth_filter=args.use_smooth_filter,
                transition_width=args.transition_width,
                recovery_method=args.recovery_method,
                recovery_strength=args.recovery_strength
            )
                    
        # Evaluate baseline
        CA = train_test.test(net, testloader)
        ASR = train_test.test(net, poisoned_testloader)
        
        print(f"Baseline - CA: {CA:.4f}, ASR: {ASR:.4f}")
        
        
        original_ca, _, _, _, _ = evaluate_two_stage_defense(
            args,
            net,
            testloader,
            clean_targets,
            target_label=args.t_b,
            sr_model=swinir_model,  # SwinIR model
            device=device,
            N=args.n_eval,
            min_scale=args.min_scale,
            num_bands=args.num_bands,
            use_smooth_filter=args.use_smooth_filter,
            transition_width=args.transition_width,
            recovery_method=args.recovery_method,
            recovery_strength=args.recovery_strength,
            verbose_per_image=args.verbose_per_image
        )
        
        
        # Evaluate two-stage defense (MODIFIED: Now uses SwinIR)
        clean_acc, filtered_asr, stage0_count, stage1_count, stage2_count = evaluate_two_stage_defense(
            args,
            net,
            poisoned_testloader,
            clean_targets,
            target_label=args.t_b,
            sr_model=swinir_model,  # SwinIR model
            device=device,
            N=args.n_eval,
            min_scale=args.min_scale,
            num_bands=args.num_bands,
            use_smooth_filter=args.use_smooth_filter,
            transition_width=args.transition_width,
            recovery_method=args.recovery_method,
            recovery_strength=args.recovery_strength,
            verbose_per_image=args.verbose_per_image
        )
        
        # Store results
        result = {
            'Dataset': args.dataset,
            'Attack': args.atk,
            'Baseline_CA': CA,
            'Baseline_ASR': ASR,
            'Defense_CA' : original_ca,
            'Defense_PA': clean_acc,
            'Defense_ASR': filtered_asr,
            'ASR_Reduction': ASR - filtered_asr
        }
        results.append(result)
        
        # Print result immediately
        print("\n" + "-"*100)
        print(f"RESULT: {args.dataset.upper()} - {args.atk.upper()} (Two-Stage Defense with SwinIR)")
        print("-"*100)
        print(f"Baseline CA:       {CA:.4f}")
        print(f"Baseline ASR:      {ASR:.4f}")
        print(f"Defense CA:        {original_ca:.4f}")
        print(f"Defense PA:        {clean_acc:.4f}")
        print(f"Defense ASR:       {filtered_asr:.4f}")
        print(f"ASR Reduction:     {ASR - filtered_asr:.4f}")
        print(f"Stage 1 Usage:     {stage1_count}/{args.n_eval} ({stage1_count/args.n_eval*100:.1f}%)")
        print(f"Stage 2 Usage:     {stage2_count}/{args.n_eval} ({stage2_count/args.n_eval*100:.1f}%)")
        print("-"*100 + "\n")
        
        # Clear memory
        del poisoned_testloader, poisoned_dataset, base_test_dataset
        gc.collect()
        torch.cuda.empty_cache()


# Print final results
results_df = pd.DataFrame(results)

print("\n" + "="*100)
print("FINAL RESULTS SUMMARY - TWO-STAGE DEFENSE PIPELINE (with SwinIR)")
print("="*100)
print(results_df.to_string(index=False))
print("\n" + "="*100)

# Save results to CSV
results_df.to_csv(f'two_stage_defense_swinir_results.csv', index=False)
print(f"\nResults saved to: two_stage_defense_swinir_results.csv")
# Lite-BD: A Two-Stage Backdoor Defense Framework

Lite-BD is a lightweight backdoor defense framework for image classification models. It uses a two-stage pipeline to neutralize backdoor triggers at inference time without requiring retraining or access to the original training data.

## Overview

Backdoor attacks embed hidden triggers into deep neural networks during training, causing the model to misclassify any input containing the trigger to an attacker-chosen target class. Lite-BD defends against these attacks by:

1. **Stage 1 — Random Resize & Pad with Super-Resolution Recovery**: The input image is randomly downscaled and padded with zeros, which disrupts spatially-localized triggers. A super-resolution model then restores the image to its original size and quality.

2. **Stage 2 — Frequency Band Detection**: If Stage 1 does not change the model's prediction, Lite-BD systematically scans frequency bands using 2D FFT to locate the band carrying the trigger signal. The identified band is removed via a bandstop filter, and the image is restored using one of several recovery methods.

If neither stage changes the prediction, the original image is returned unmodified.

## Supported Attacks

| Attack | Type |
|--------|------|
| BadNet | Patch-based |
| Blend | Blending-based |
| WaNet | Warping-based |
| SIG | Sinusoidal signal |
| CL (Clean Label) | Patch-based, clean label |
| BPPAttack | Dithering/compression |
| Trojan | Image overlay |
| LF | Low-frequency perturbation |
| Poison Ink | Edge injection |
| LIRA | Learned trigger |
| FIBA | Frequency-based |
| Refool | Reflection-based |
| Filter | Instagram-style filter |

## Supported Datasets & Models

| Dataset | Model |
|---------|-------|
| CIFAR-10 | ResNet-18 |
| GTSRB | VGG-11 |
| Fashion-MNIST | ResNet-18 |

## File Structure

```
├── Lite-BD(SW).py              # Main defense script using SwinIR for Stage 1 SR
├── Lite-BD(RE).py              # Variant using Real-ESRGAN for Stage 1 SR
├── backdoor_triggers.py        # Trigger injection functions for all supported attacks
├── backdoor_triggers_extended.py  # Additional trigger (Poison Ink)
├── train_test.py               # Training and evaluation utilities
├── network_swinir.py           # SwinIR model architecture
├── resnet.py                   # ResNet-18 architecture
├── vgg.py                      # VGG architecture
├── cifar10.py                  # CIFAR-10 dataset helper
├── SR_models/                  # Pretrained super-resolution model weights
├── checkpoint/                 # Backdoored model checkpoints
├── pattern25.png               # Default trigger pattern image
├── sig.pt                      # SIG attack pattern
└── tiny_preactresnet18_0_255.npy  # LF attack trigger pattern
```

## Requirements

```
torch
torchvision
numpy
Pillow
opencv-python
scikit-learn
tqdm
pandas
matplotlib
seaborn
basicsr          # for Real-ESRGAN variant
realesrgan       # for Real-ESRGAN variant
```

Install with:

```bash
pip install torch torchvision numpy Pillow opencv-python scikit-learn tqdm pandas matplotlib seaborn
# For Real-ESRGAN variant only:
pip install basicsr realesrgan
```

## Pretrained Weights

Place pretrained SR model weights in the `SR_models/` directory:

- **SwinIR** (used in `Lite-BD(SW).py`): `SR_models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth`
- **Real-ESRGAN** (used in `Lite-BD(RE).py`): `SR_models/RealESRGAN_x4plus.pth`

Backdoored model checkpoints must be placed in `checkpoint/` following the naming convention:

```
checkpoint/{dataset}_{attack}_t_{target_label}_p_{poison_ratio}.pth
```

Example: `checkpoint/cifar10_badnet_t_3_p_10.0.pth`

## Usage

### SwinIR variant

```bash
python "Lite-BD(SW).py" \
    --dataset cifar10 \
    --atk badnet \
    --t_b 3 \
    --n_eval 100 \
    --min_scale 0.5 \
    --num_bands 50 \
    --recovery_method unsharp \
    --recovery_strength 0.5 \
    --swinir_path SR_models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth \
    --swinir_scale 2
```

### Real-ESRGAN variant

```bash
python "Lite-BD(RE).py" \
    --dataset cifar10 \
    --atk badnet \
    --t_b 3 \
    --n_eval 100 \
    --min_scale 0.5 \
    --num_bands 50 \
    --recovery_method unsharp \
    --recovery_strength 0.5
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `cifar10` | Dataset: `cifar10`, `gtsrb`, `fashion-mnist` |
| `--atk` | `badnet` | Attack type (see supported attacks above) |
| `--t_b` | `3` | Target backdoor label |
| `--p` | `1.0` | Poison ratio used during training |
| `--n_eval` | `100` | Number of images to evaluate |
| `--min_scale` | `0.5` | Stage 1 minimum resize scale (0–1) |
| `--num_bands` | `50` | Stage 2 number of frequency bands to test |
| `--recovery_method` | `unsharp` | Stage 2 recovery: `super_resolution`, `bilateral`, `unsharp`, `none` |
| `--recovery_strength` | `0.5` | Stage 2 recovery strength (0–1) |
| `--use_smooth_filter` | `True` | Use smooth frequency transitions in Stage 2 |
| `--transition_width` | `0.08` | Smooth transition band width (0.01–0.1) |
| `--verbose_per_image` | `False` | Print per-image debug info |

## Output Metrics

| Metric | Description |
|--------|-------------|
| **CA** (Clean Accuracy) | Baseline accuracy on clean test images |
| **ASR** (Attack Success Rate) | Fraction of poisoned inputs classified as target label before defense |
| **Defense CA** | Accuracy on clean images after applying defense |
| **Defense PA** (Preserved Accuracy) | Accuracy on poisoned images correctly classified after defense |
| **Defense ASR** | Fraction of poisoned inputs still classified as target label after defense |
| **ASR Reduction** | `Baseline ASR − Defense ASR` |

Results are saved to `litebd_re_results.csv` (Real-ESRGAN variant) or `two_stage_defense_swinir_results.csv` (SwinIR variant).

## How It Works

```
Input Image (poisoned)
       │
       ▼
┌─────────────────────────────────┐
│  Stage 1: Resize & Pad + SR     │
│  - Downsample by min_scale      │
│  - Pad to original size         │
│  - Upsample with SwinIR/ESRGAN  │
└─────────────────────────────────┘
       │
  Label changed?
   YES ──► Return Stage 1 output
       │
      NO
       │
       ▼
┌─────────────────────────────────┐
│  Stage 2: Frequency Band Filter │
│  - FFT → test each band         │
│  - Find band that flips label   │
│  - Apply bandstop filter        │
│  - Apply recovery method        │
└─────────────────────────────────┘
       │
  Label changed?
   YES ──► Return Stage 2 output
       │
      NO
       │
       ▼
  Return original image
```

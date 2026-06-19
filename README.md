# Lite-BD: A Lightweight Black-box Backdoor Defense via Reviving Multi-Stage Image Transformations

Deep Neural Networks (DNNs) are vulnerable to backdoor attacks. Due to the nature of Machine Learning as a Service (MLaaS) applications, black-box defenses are more practical than white-box methods, yet existing purification techniques suffer from key limitations: a lack of justification for specific transformations, dataset dependency, high computational overhead, and a neglect of frequency-domain transformations. This paper conducts a preliminary study on various image transformations, identifying down-upscaling as the most effective backdoor trigger disruption technique. We subsequently propose \texttt{Lite-BD}, a lightweight two-stage blackbox backdoor defense. \texttt{Lite-BD} first employs a super-resolution-based down-upscaling stage to neutralize spatial triggers. A secondary stage utilizes query-based band-by-band frequency filtering to remove triggers hidden in specific bands. Extensive experiments against state-of-the-art attacks demonstrate that \texttt{Lite-BD} provides robust and efficient protection. 

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

## Supported Datasets & Models

| Dataset | Model |
|---------|-------|
| CIFAR-10 | ResNet-18 |
| GTSRB | VGG-11 |
| Fashion-MNIST | ResNet-18 |

To downalod the CIFAR-10 dataset and arrange it as needed please run the following:

```bash
python "cifar10.py" 
```

For GTSRB and Fashion-Mnist please download the dataset from following link and unzipt it:

GTSRB and Fashion-Mnist: https://drive.google.com/file/d/1i22oFGZ3mRmPG69VG8_JN48LGSqKzOH7/view?usp=sharing

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
├── SR_models/                  # Pretrained super-resolution model weights (needs to be downloaded)
├── checkpoint/                 # Backdoored model checkpoints (needs to be downloaded)
├── pattern25.png               # Default trigger pattern for Badnet
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

To download the SR model weights and backdoored model checkpoints please download the zip filed from: https://drive.google.com/file/d/1XnwUSRV-9tJlUaKhEGKU4Ks9RkuN3s3C/view?usp=sharing

Unzip the zip file and it will contain two folders 'SR_models' and 'checkpoint'. 

## Usage

### SwinIR variant

```bash
python "Lite-BD(SW).py" 
```

### Real-ESRGAN variant

```bash
python "Lite-BD(RE).py" 
```

Both of the codes will evaluate Lite-BD on all ten attacks for all three datasets. If you want to see the results for a specific attack or dataset please modify following lines from Lite-BD(SW) and Lite-BD(RE):

attacks = ['badnet', 'blend', 'wanet', 'sig', 'cl', 'bppattack', 'trojan', 'lf',  'poison-ink', 'lira']
datasets = ['cifar10', 'gtsrb', 'fashion-mnist']



## Output Metrics

| Metric | Description |
|--------|-------------|
| **CA** (Clean Accuracy) | Baseline accuracy on clean test images |
| **ASR** (Attack Success Rate) | Fraction of poisoned inputs classified as target label before defense |
| **Defense CA** | Accuracy on clean images after applying defense |
| **Defense PA** (Preserved Accuracy) | Accuracy on poisoned images correctly classified after defense |
| **Defense ASR** | Fraction of poisoned inputs still classified as target label after defense |
| **ASR Reduction** | `Baseline ASR − Defense ASR` |

Results are saved to `litebd_re_results.csv` (Real-ESRGAN variant) or `litebd_sw_results.csv` (SwinIR variant).

## Citation

If you find our work insight or useful, please consider citing:

```bash
@article{miah2026lite,
  title={Lite-BD: A Lightweight Black-box Backdoor Defense via Reviving Multi-Stage Image Transformations},
  author={Miah, Abdullah Arafat and Bi, Yu},
  journal={arXiv preprint arXiv:2602.07197},
  year={2026}
}
```

## Contact

For any question regarding the repo or paper please contact: abdullaharafat.miah@uri.edu or yu_bi@uri.edu

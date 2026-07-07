# Blur2Sun: Self-Supervised Solar EUV Image Deconvolution via Deep Unfolding

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Blur2Sun** is a self-supervised PSF deconvolution framework for SDO/AIA solar extreme ultraviolet (EUV) imagery. It couples the [Blur2Blur](https://github.com/VinAIResearch/blur2blur) self-supervision paradigm with a BID-inspired deep unfolding architecture, embedding the wavelength-dependent AIA PSF as a differentiable forward operator. No sharp ground truth is required.

> 📄 **Paper**: *Submitted to APJS* (under review)

---

## Key Features

- **Self-supervised**: Trains on temporal adjacent AIA frame pairs — no need for sharp ground truth
- **Physics-informed**: Embeds the revised [Hofmeister et al. (2025)](https://doi.org/10.1051/0004-6361/202451802) multi-channel AIA PSF as a differentiable FFT-based forward operator
- **Deep Unfolding Architecture (SDUN)**: 10 learnable BIDBlocks, each coupling explicit PSF degradation/conjugate back-projection with a learnable regularizer
- **Multi-channel**: Supports all 7 AIA EUV passbands (94, 131, 171, 193, 211, 304, 335 Å)
- **Comprehensive evaluation**: Lunar-transit stray-light tests, noise robustness, FWHM/PSD analysis, DEM inversions, Hi-C 2.1 cross-validation, ablation study

## Architecture Overview

```
Y₁ ──→ [BIDBlock 1] → [BIDBlock 2] → ... → [BIDBlock 10] → Softplus⁺ → X̂₁
         │                │                      │
         ├ PSF Degrade    ├ PSF Degrade          ├ PSF Degrade
         ├ PSF Adj Back   ├ PSF Adj Back         ├ PSF Adj Back
         └ DP Prior       └ DP Prior             └ DP Prior

Loss: L₁(Y₂, H(X̂₁)) + L₁(Y₁, H(X̂₂))   (bidirectional reblurring loss)
```

Each BIDBlock structurally couples:
1. **PSF Forward Degradation** — FFT‑based convolution with the channel‑specific AIA PSF
2. **Conjugate Back‑Projection** — adjoint PSF operator for gradient consistency
3. **DP Prior** — lightweight 4‑layer CNN regularizer (GroupNorm + LeakyReLU) that learns to recover high‑frequency coronal structure without amplifying noise

The step size and prior correction are learned from data rather than hand‑tuned.

## Requirements

- Python ≥ 3.10
- [PyTorch](https://pytorch.org/) ≥ 2.1 (CUDA recommended)
- [Astropy](https://www.astropy.org/) — FITS I/O, coordinate handling
- [Sunpy](https://sunpy.org/) — AIA colormaps, solar coordinates
- [SciPy](https://scipy.org/) — FFT, interpolation
- [scikit-image](https://scikit-image.org/) — image processing
- [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/)
- [easydict](https://pypi.org/project/easydict/)

Install dependencies:

```bash
pip install torch torchvision numpy scipy matplotlib astropy sunpy scikit-image easydict
```

## Project Structure

```
Blur2Sun/
├── config.py                    # Main configuration (paths, model, training, PSF params)
├── model.py                     # SDUN model: BIDBlock + DP Prior + Softplus⁺
├── data_loader.py               # AIA FITS data loader with pair filtering
├── data_utils.py                # File sorting, pair generation, quality filtering
├── loss.py                      # Bidirectional L₁ reblurring loss
├── metrics.py                   # PSNR, RMSE, SSIM, flux error metrics
├── deconv.py                    # BID & Richardson-Lucy deconvolution (CPU/GPU)
├── visualization.py             # Plotting: PSD, profiles, DEM, comparison figures
├── train.py                     # Training script (full model, all bands)
├── train_ablation.py            # Ablation training (module / T iterations / step size)
├── evaluate.py                  # Main evaluation: FWHM, PSD, zoom comparison, Luna transit
├── evaluate_ablation.py         # Ablation evaluation (94 Å, inverse consistency)
├── PSF/                         # AIA PSF FITS files (psf_aia_0XXX.fits)
├── data/                        # AIA Level 1.5 data (not included)
│   ├── classified_aia_fits/     # Flare dataset (training)
│   │   ├── 0094-HR/
│   │   ├── 0131-HR/
│   │   └── ...
│   ├── classified_aia_bands_fits/ # Quiet-Sun dataset
│   └── test/test_classified/    # Test dataset
└── outputs/                     # Model checkpoints, logs, results
```

## Data Preparation

We use SDO/AIA Level 1.5 FITS files from [JSOC](http://jsoc.stanford.edu/). The data should be organized by wavelength:

```
data/
├── classified_aia_fits/          # Training: flare events
│   ├── 0094-HR/                  # 94 Å channel
│   └── ...
└── test/test_classified/         # Testing
    ├── 0094-HR/
    └── ...
```

### PSF Files

Download the revised multi-channel AIA PSF models from [Hofmeister et al. (2025)](https://doi.org/10.1051/0004-6361/202451802) and place them in `./PSF/`:

```
PSF/
├── psf_aia_0094.fits
├── psf_aia_0131.fits
├── psf_aia_0171.fits
├── psf_aia_0193.fits
├── psf_aia_0211.fits
├── psf_aia_0304.fits
└── psf_aia_0335.fits
```

### Configuration

Edit `config.py` to set your data paths:

```python
cfg.BASE.FLARE_DATA_ROOT = "./data/classified_aia_fits/"
cfg.BASE.QUIET_DATA_ROOT = "./data/classified_aia_bands_fits/"
cfg.BASE.OUTPUT_ROOT = "./outputs/"
```

PSF file paths are configured per-channel in the `aia_band_params` dictionary.

## Usage

### Training

Train the full model on all 7 AIA EUV channels (each channel independently):

```bash
python train.py
```

For ablation study (training without the DP Prior):

```bash
python train_ablation.py
```

Training details:
- **Hardware**: NVIDIA RTX 5090 D (32 GB VRAM) or equivalent
- **Epochs**: 50 per channel
- **Batch size**: 1
- **Optimizer**: AdamW (lr=1e-4, weight decay=1e-5, cosine annealing to 1e-7)
- **Loss**: Bidirectional L₁ reblurring loss
- **Early stopping**: patience=5 on validation loss
- **Seed**: 42

### Evaluation

Run the main evaluation pipeline (FWHM, PSD, zoom comparisons, lunar transit):

```bash
python evaluate.py
```

Run ablation evaluation (94 Å only):

```bash
python evaluate_ablation.py
```

### Standalone Deconvolution

Apply BID or Richardson-Lucy deconvolution to individual images:

```python
from deconv import deconvolve_bid, deconvolve_richardson_lucy
from astropy.io import fits

img = fits.getdata("your_image.fits")
psf = fits.getdata("PSF/psf_aia_0171.fits")

# BID deconvolution
bid_result = deconvolve_bid(img, psf, iterations=10)

# Richardson-Lucy deconvolution
rl_result = deconvolve_richardson_lucy(img, psf, iterations=10)
```

## Results

Evaluation across six complementary angles:

| Evaluation Angle | Key Finding |
|---|---|
| **Lunar Transit** | Zero-baseline convergence inside occulted lunar disk across all 7 channels |
| **Noise Robustness** | Blur2Sun outperforms RL and BID at all tested SNR levels |
| **FWHM / PSD** | Physically consistent FWHM reduction without noise amplification; PSD closest to observed |
| **DEM Inversion** | Consistent thermodynamic recovery for both flare and quiet-Sun regions |
| **Hi-C 2.1 Cross-Validation** | Closely reproduces Hi-C sub-arcsecond intensity profiles |
| **Ablation (DP Prior)** | +2.36 dB PSNR gain; flux error reduced from 10.7% to 2.4% |

## Citation

If you use Blur2Sun in your research, please cite:

```bibtex
@article{zhao2025blur2sun,
  title={Blur2Sun: Self-Supervised Solar EUV Image Deconvolution via Deep Unfolding},
  author={Zhao, Tianhao and Qiang, Zhenping and others},
  journal={The Astrophysical Journal Supplement Series},
  year={2025},
  note={Under review}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SDO/AIA data courtesy of NASA/SDO and the AIA science team (JSOC)
- PSF models from [Hofmeister et al. (2025)](https://doi.org/10.1051/0004-6361/202451802) (NASA grant 80NSSC20K0183)
- Blur2Blur paradigm inspired by [Pham et al. (2024)](https://arxiv.org/abs/2404.10739)
- BID algorithm from [Hofmeister et al. (2024)](https://www.aanda.org/articles/aa/full_html/2024/09/aa50245-24/aa50245-24.html)

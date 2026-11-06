# Blur2Sun: Self-Supervised Solar EUV Image Deconvolution via Deep Unfolding

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Blur2Sun** is a self-supervised PSF deconvolution framework for SDO/AIA solar extreme ultraviolet (EUV) imagery. It couples the [Blur2Blur](https://github.com/VinAIResearch/blur2blur) self-supervision paradigm with a BID-inspired deep unfolding architecture, embedding the wavelength-dependent AIA PSF as a differentiable forward operator. No sharp ground truth is required.

---

## Key Features

- **Self-supervised**: trains on temporal adjacent AIA frame pairs (12 s cadence, screened by signal-region NCC with a per-band/per-day median − 4×MAD threshold) — no sharp ground truth needed
- **Physics-informed**: embeds the official `rebin_psf`-rebinned AIA PSF (8192²→2048², half-pixel centre corrected) as a differentiable FFT forward operator — the PSF array is twice the image size, as required for full-field scattering
- **Deep unfolding (v5)**: 10 unrolled restoration blocks (RBs), each coupling a **BID-form image-space residual-feedback data step** (learnable step size initialised at 1, per-iteration positivity constraint — the same residual update as BID, Hofmeister et al. 2024) with a **learned prior gradient** (PriorNet, additive correction)
- **Multi-channel**: all 7 AIA EUV passbands (94, 131, 171, 193, 211, 304, 335 Å)
- **Ready to run**: a one-command demo on bundled 3.6 MB test samples, plus a single full-disk 7-band × 4-column visualisation script as the lightweight inference example

## Architecture Overview

```
Y₁ ──→ [RB 1] → [RB 2] → ... → [RB 10] → X̂₁
   │                    │
   ├ BID-form residual:  x ← clamp(x − k·(P(x) − y), 0)   k learnable (init 1)
   ├ prior:              x ← x_lw − β·PriorNet(x_lw)      β learnable
   └ PSF operator P (differentiable FFT convolution)

Loss: L₁(Y₂, P(X̂₁)) + L₁(Y₁, P(X̂₂))  (bidirectional reblurring loss)
```

Each restoration block (RB; implemented as `MAPBlock` in `src/model2_map.py`) couples:

1. **BID-form residual data step** — the image-space residual-feedback update with a learnable per-iteration step size (the network learns an annealing schedule, e.g. k ≈ 1.4 → 0.2 across the 10 iterations at 171 Å, consistent with the semi-convergence behaviour of iterative deconvolution)
2. **Learned prior gradient (PriorNet)** — lightweight 4-layer CNN regularizer (GroupNorm + LeakyReLU) applied as an additive correction

## Repository Structure

```
├── configs/                  # training configurations
├── src/
│   ├── model2_map.py         # SDUN network: restoration blocks (BID-form data step + PriorNet)
│   ├── loss.py               # Blur2Blur bidirectional reblurring loss
│   ├── data_loader.py        # 12s paired dataset (NCC + 4xMAD screening)
│   ├── data_utils.py         # FITS IO / preprocessing
│   ├── metrics.py            # evaluation metrics
│   └── deconvolve_image.py   # official BID implementation (Hofmeister 2024)
├── training/train_map.py     # training entry point
├── evaluation/
│   ├── test_proposed_v5.py   # full-disk 7x4 visualisation (inference example)
│   └── paper_tables_v5/v5_common.py  # model loader + PSF/RL/BID helpers
├── scripts/make_release_bundle.sh  # rebuild the weight Release assets
├── demo/                     # one-command demo + bundled test samples
├── psf/                      # rebinned AIA PSFs (2048², per channel)
├── paper_figures/            # paper figures + figure↔section map
└── docs/                     # technical report & verification notes
```

## Requirements

```
pip install -r requirements.txt
```

Python ≥ 3.10, PyTorch ≥ 2.1 (CUDA recommended), plus astropy, sunpy, scipy, matplotlib. See `requirements.txt` for the full list.

## Data

Full training/evaluation archives are **not** included (SDO/AIA and Hi-C data are publicly available from JSOC), but **compact representative test samples ship with the repository** in `demo/demo_data/` so the example figures can be reproduced without any data download.

The scripts expect the following layout, configurable via the `BLUR2SUN_DATA` environment variable (default `/mnt/zth24029/DataHDD/Hi-C/Blur_data`):

```
$BLUR2SUN_DATA/
├── paired_flare_20240515/    # 12s training pairs (flare day)
└── paired_quiet_20210218/    # 12s training pairs (quiet day)
```

The 7×4 visualisation additionally reads a set of test frames via `BLUR2SUN_TEST` (default `$BLUR2SUN_DATA/test/test_classified`).

### Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `BLUR2SUN_MODELS` | directory containing `band_XX/map_full_12s_v5/models/best_model.pth` | `<repo>/outputs` |
| `BLUR2SUN_DATA` | root of the JSOC-derived training archives | `/mnt/zth24029/DataHDD/Hi-C/Blur_data` (override on another machine) |
| `BLUR2SUN_TEST` | test frames for the 7×4 visualisation | `$BLUR2SUN_DATA/test/test_classified` |
| `BLUR2SUN_SHORT_EXPOSURE` | set to `1` to apply the short-exposure input offset in the 7×4 script | `0` (native intensity) |
| `JSOC_EMAIL` | your registered JSOC email (never hard-coded) | set before any JSOC download |

## Usage

### Training

```
# full model, band 171
python3 training/train_map.py --bands 171 --config configs/config_map_12s_v5.py --gpu 0
```

### Inference example: full-disk 7×4 visualisation

```
export BLUR2SUN_MODELS=/path/to/outputs
export BLUR2SUN_TEST=/path/to/test_classified
python3 evaluation/test_proposed_v5.py
# -> AIA_7BANDS_7x4_VISUALIZATION.png (7 channels x 4 columns)
```

This is the only evaluation/inference example shipped with the repository; it loads the released checkpoints through `evaluation/paper_tables_v5/v5_common.py` (model loader with checkpoint-key and module-source guards) and visualises the full-disk observation, the Blur2Sun output, and their relative change.

### Pre-trained weights (Google Drive)

The trained model weights are hosted on Google Drive:

**[Blur2Sun v5 weights (Google Drive)](https://drive.google.com/drive/folders/1UUtPrxP9i2zw6udOLVLXMKcx9VN0khwV?usp=drive_link)** — 7 passbands, `band_XX/map_full_12s_v5/models/best_model.pth` — the checkpoints used for all main-text results.

```
# after downloading and unzipping
export BLUR2SUN_MODELS=/path/to/outputs
# contains band_XX/map_full_12s_v5/models/best_model.pth
```

### Inference configuration (input offset)

Standard-exposure frames are deconvolved at native intensity with **no input offset** (the default everywhere in this repository). For count-depleted short-exposure flare frames only, a **hand-set per-channel calibration constant** (2, 2, 85, 4, 85, 4, 4 DN for 94/131/171/193/211/304/335 Å) is added to the input and subtracted from the output; see Section 4.4 of the paper. `demo/run_demo.py` applies this rule per frame, and the 7×4 visualisation can enable it with `BLUR2SUN_SHORT_EXPOSURE=1`.

### Quick demo (lightweight, ~3.6 MB data)

```
export BLUR2SUN_MODELS=/path/to/outputs   # v5 weights
python3 demo/run_demo.py
# -> demo/output/*.png
```

`demo/demo_data/` contains compact test samples (X9.3 flare frame at 94 Å, quiet-Sun frame at 171 Å, lunar-edge profiles for 7 channels) so the example figures can be reproduced immediately once the weights are available.

### Efficiency (1024x1024 frame, 171 A)

| Method | Iterations | Device | Time/frame |
| --- | --- | --- | --- |
| Blur2Sun v5 (production recipe) | 10 blocks | GPU | **0.19 s** |
| RL (corrected PSF) | 30 | GPU | 0.39 s |
| BID (official implementation) | 30 | CPU | 0.95 s |

## License

MIT (see [LICENSE](LICENSE)). The BID implementation in `src/deconvolve_image.py` follows the official BID implementation of Hofmeister et al. (2024).

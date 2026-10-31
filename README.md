# Fused Zip-NeRF
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=PyTorch&logoColor=white)&nbsp;
![CUDA](https://img.shields.io/badge/-CUDA-76B900?logo=NVIDIA&logoColor=white)&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

__A fully fused CUDA implementation of [Zip-NeRF](https://jonbarron.info/zipnerf/) that reaches the quality of the official implementation while training approximately 6-7x and rendering approximately 10x faster.__

This project reimplements the following paper, closely following its official JAX implementation
([camp_zipnerf](https://github.com/jonbarron/camp_zipnerf)). Consider a citation when applicable.

```bibtex
@inproceedings{barron2023zipnerf,
  title     = {{Zip-NeRF}: Anti-Aliased Grid-Based Neural Radiance Fields},
  author    = {Barron, Jonathan T. and Mildenhall, Ben and Verbin, Dor and Srinivasan, Pratul P. and Hedman, Peter},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  pages     = {19640--19648},
  doi       = {10.1109/ICCV51070.2023.01804},
  url       = {https://jonbarron.info/zipnerf/}
}
```


## Overview

Zip-NeRF combines the multisampled, anti-aliased ray model of [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) with
the multiresolution hash grids of [Instant-NGP](https://nvlabs.github.io/instant-ngp/). Each ray is sampled by two
proposal levels and a final NeRF level, each with its own hash grid. This implementation replaces the official
JAX/autograd pipeline with hand-written CUDA kernels and analytic gradients, without tiny-cuda-nn or PyTorch autograd:

- **Ray generation** (`sampling/`): the pixels of all training views live on the GPU, and the rays of each batch are
  generated on the fly. All random numbers (pixel selection, sampling jitter, multisampling patterns, background colors)
  are counter-based hashes of the iteration's seed, so no kernel stores or reads them.
- **Featurization** (`featurization/`): the multisampled hash grid lookups (hexagonal multisampling of conical
  frustums, scene contraction, and feature downweighting) run in runtime-compiled kernels (NVRTC) with the grid layout
  baked in as constants. The small proposal MLPs are fused into these kernels in fp32, in the forward as well as in the
  backward pass.
- **NeRF MLP** (`mlp/`): a chain of fp16 cuBLASLt GEMMs with fused bias and ReLU epilogues. The bottleneck only feeds
  linear layers and is therefore folded into their weights.
- **Losses** (`sampling/`): compositing, resampling, and all losses (Charbonnier data loss, interlevel losses with
  blurred resampling, and distortion loss) are computed together with their gradients in fused per-ray kernels.
- **Optimizer** (`optimizer/`): a single-launch Adam over all parameter buffers that also applies the grid
  regularizer and updates the magnitudes of the hash grids' scale features.

Inference uses the same kernels with deterministic sampling. Rays are processed in Morton order and each warp handles
the same interval of 32 neighboring rays, so their grid lookups mostly hit the same cache lines.


## Installation

This method is provided as an extension to [NeRFICG](https://github.com/nerficg-project), a radiance field and
view synthesis framework.

### Requirements

- An NVIDIA GPU
- Linux or Windows (Windows builds should work but are untested)
- A recent CUDA SDK ([CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) recommended) and a compatible C++ compiler
- [Anaconda / Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) installed

### Setup

As a preparatory step, the [NeRFICG framework](https://github.com/nerficg-project/nerficg) needs to be set up.
Please follow the instructions in its README to set up a compatible Conda environment.

Now add Zip-NeRF as an additional method by cloning this repository into `src/Methods/ZipNeRF`:
```shell
# HTTPS
git clone https://github.com/fhahlbohm/fused-zipnerf.git src/Methods/ZipNeRF
```
or
```shell
# SSH
git clone git@github.com:fhahlbohm/fused-zipnerf.git src/Methods/ZipNeRF
```

Next, install all method-specific dependencies and CUDA extensions using:
```shell
python ./scripts/install.py -m ZipNeRF
```

_Note: The framework determines on-the-fly what extra modules need to be installed. Sometimes this causes unnecessary errors/warnings that can interrupt the installation process. In this case, first try to rerun the command before investigating the error in detail._

The featurization kernels are compiled at runtime from the sources in this repository on first use and cached in
`ZipNeRFCudaBackend/rtc/cache`. As the CUDA backend reads these sources from the repository, it cannot be used as a
standalone PyTorch extension.


## Usage

The provided config [`zipnerf_garden.yaml`](zipnerf_garden.yaml) trains the garden scene of the Mip-NeRF 360 dataset:
```shell
python ./scripts/train.py -c src/Methods/ZipNeRF/zipnerf_garden.yaml
```
Afterwards, the NeRFICG script `inference.py` can provide quality metrics, FPS benchmarking, etc. as described in the
[NeRFICG repository](https://github.com/nerficg-project/nerficg).

The default settings follow the training protocol of the paper: batches of 65536 rays, 25k iterations, and learning
rates from 1e-2 to 1e-3. The paper trained on 8 GPUs, the official code release instead provides a single-GPU config
with batches of 8192 rays, 200k iterations, and 8x lower learning rates (1.25e-3 to 1.25e-4, 20k warmup iterations).
As this implementation processes 65536 rays at once on a single GPU, the paper protocol is the faster option.

The `MODEL` section defines the samples per level, the hash grids, and the MLP widths of the
[official 360 config](https://github.com/jonbarron/camp_zipnerf/blob/main/configs/zipnerf/360.gin). In the `RENDERER` section, `RENDER_EXTRAS` additionally outputs the opacity and the depth during
inference (off by default, as it slows down rendering). The depth is the expected depth as in the official code, or the
median depth if `USE_MEDIAN_DEPTH` is enabled.


## Results

So far, we have only run limited tests. Garden scene of the Mip-NeRF 360 dataset (the dataset's pre-downscaled JPEGs in `images_4`, every 8th image held out for testing), default config:

| | PSNR | SSIM |
|---|---|---|
| Zip-NeRF paper | 28.20 | 0.860 |
| Official code release | 28.22 | 0.863 |
| This implementation | 28.22 | 0.860 |

All SSIM values use the SSIM implementation of the official code release (`dm_pix.ssim`).

Training takes 35:47 min (about 10 GB VRAM) and rendering a 1297x840 test view takes about 300 ms on an RTX 5090. For
comparison, the official implementation takes about 5 hours to train this scene and renders at 0.1-0.2 FPS on an RTX 4090.

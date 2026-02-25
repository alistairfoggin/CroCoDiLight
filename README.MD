# [ICLR 2026] CroCoDiLight: Repurposing Cross-View Completion Encoders for Relighting

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://alistairfoggin.com/projects/crocodilight)
[![Paper](https://img.shields.io/badge/OpenReview-Paper-red)](https://openreview.net/forum?id=GKvb3HCyNk)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/alistairfoggin/CroCoDiLight)

Disentangles illumination from scene content in CroCo (Cross-view Completion) latent representations. A learned lighting
extractor separates each encoder embedding into a single lighting vector and lighting-invariant patch features, which
can then be recombined with target lighting conditions. This enables shadow removal, albedo estimation, lighting
transfer, and interpolation, trained on datasets two orders of magnitude smaller than the original CroCo pretraining.

## Setup

Create a conda environment:

```bash
conda create -n crocodilight python=3.10 -y
conda activate crocodilight
conda install pytorch torchvision -c pytorch
```

> NOTE: croco (and optionally curope) are installed as standalone packages in this conda environment from the
`croco_module` branch of the CroCo repository.

Install with curope CUDA kernels for faster RoPE positional embeddings. You can install `cuda-toolkit`
instead of `cuda-cudart-dev` and `cuda-nvcc` if you prefer a complete CUDA installation. A system CUDA
installation should also work but has not been tested. The curope CUDA kernels compile for all
architectures and may take some time. To target a specific architecture, set `TORCH_CUDA_ARCH_LIST`.
If you have a custom CUDA installation, set `CUDA_HOME` accordingly:

```bash
conda install cuda-cudart-dev cuda-nvcc ninja -c nvidia
pip install -e ".[curope]" --no-build-isolation
```

Or install without curope (a pure PyTorch fallback is used automatically):

```bash
pip install -e .
```

To run the gradio demos, you also need to run the following:

```bash
pip install -e ".[demos]"
```

## Pretrained Models

All weights should be placed in `pretrained_models/`.

### For inference

| File                             | Required for          | Description                                                                                                           |
|----------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------|
| `CroCoDiLight.pth`               | All inference scripts | Full CroCoDiLight model (includes the CroCo encoder, single-view decoder, lighting extractor, and lighting entangler) |
| `CroCoDiLight_shadow_mapper.pth` | Shadow removal        | Lighting mapper trained for shadow removal                                                                            |
| `CroCoDiLight_albedo_mapper.pth` | Albedo estimation     | Lighting mapper trained for intrinsic image decomposition                                                             |

`CroCoDiLight.pth` is the base model needed by every inference and evaluation script. The mapper weights are only needed
for their respective tasks (shadow removal or albedo estimation). Lighting swap, freeze, transfer, and interpolation use
the base model only.

### For training

| File                                | Used in     | Description                                                                                                                                                                                                                                                                           |
|-------------------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CroCo_V2_ViTLarge_BaseDecoder.pth` | Steps 1 & 2 | Original CroCo v2 ViT-Large weights from [Weinzaepfel et al.](https://github.com/naver/croco) (includes the cross-view decoder, though only the encoder is used). Download from [here](https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth). |
| `CroCoDiLight_decoder.pth`          | Step 2      | Single-view decoder pretrained on ImageNet reconstruction (output of training step 1). Available for download but not needed for inference -- its weights are embedded in `CroCoDiLight.pth`.                                                                                         |

## Inference

All inference scripts accept `--help` for full usage and `--device` for GPU selection. Input can be a single image or a
folder.

### Gradio demos

Interactive demos for shadow removal, albedo estimation, and relighting are available
as a tabbed Gradio app:

```bash
python demos/app.py
```

Individual demos can also be run standalone (e.g. `python demos/shadow_removal.py`). Pass
`--share` to create a public link.

### Shadow removal

```bash
python scripts/inference/shadow_removal.py --input <image_or_folder> --output <image_or_folder>
```

### Albedo estimation

```bash
python scripts/inference/albedo_estimation.py --input <image_or_folder> --output <image_or_folder>
```

### Swap lighting between two images

```bash
python scripts/inference/swap_lighting.py --image1 a.png --image2 b.png --output-dir out/
```

### Freeze lighting (fixed lighting, varying content)

```bash
python scripts/inference/freeze_lighting.py --reference ref.png --input <folder> --output <folder>
```

### Transfer lighting (fixed content, varying lighting)

```bash
python scripts/inference/transfer_lighting.py --reference ref.png --input <folder> --output <folder>
```

### Interpolate lighting between two frames

```bash
python scripts/inference/interpolate_lighting.py --frame-a a.png --frame-b b.png --output-dir out/ --steps 5
```

## Training

Training requires [Weights & Biases](https://wandb.ai) for logging (`wandb login`). See
[datasets/DATASETS.md](datasets/DATASETS.md) for required datasets and expected directory layouts.

### Step 1: Pretrain the single-view decoder

```bash
python scripts/training/pretrain_relight_decoder.py
```

### Step 2: Train the relighting model

```bash
python scripts/training/train_relight_model.py
```

### Step 3: Train task-specific mappers

Edit the mapper type (`"shadow"` or `"albedo"`) at the bottom of the script, then run:

```bash
python scripts/training/train_lighting_mapper.py
```

## Evaluation

### Shadow removal

Run inference on each shadow dataset, then compute metrics (MAE, RMSE, LPIPS, PSNR, SSIM):

```bash
python scripts/inference/shadow_removal.py --input ./datasets/SRD/test/shadow/ --output ./outputs/SRD/
python scripts/inference/shadow_removal.py --input ./datasets/ISTD+/test/test_A/ --output ./outputs/ISTD+/
python scripts/inference/shadow_removal.py --input ./datasets/WSRD+/val/input/ --output ./outputs/WSRD+/
python scripts/inference/shadow_removal.py --input ./datasets/INS/test/origin/ --output ./outputs/INS/

python scripts/evaluation/shadow_metrics.py --predictions ./outputs --datasets-root ./datasets
```

The evaluation script can also be used to evaluate other methods by pointing `--predictions` at any folder
with the same structure (`SRD/`, `ISTD+/`, `WSRD+/`, `INS/` subfolders containing predicted shadow-free images).

See [datasets/DATASETS.md](datasets/DATASETS.md) for download links and expected directory layouts.

### Albedo estimation (IIW)

Run inference and compute WHDR in one pass:

```bash
python scripts/evaluation/iiw_predict_and_score.py --iiw-root ./datasets/IIW/
```

Or separately, first predict reflectance images, then score them:

```bash
python scripts/inference/albedo_estimation.py --input ./datasets/IIW/ --output ./datasets/IIW/
python scripts/evaluation/albedo_whdr.py --iiw-root ./datasets/IIW/
```

## Citation BibTeX

If you use CroCoDiLight in your research, please cite:

```bibtex
@inproceedings{foggin2026crocodilight,
  title={{CroCoDiLight}: Repurposing Cross-View Completion Encoders for Relighting},
  author={Foggin, Alistair J and Smith, William A P},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=GKvb3HCyNk}
}
```

## License

This project, including its source code and pretrained model weights, is
licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The
pretrained weights are additionally subject to the license terms of the upstream training
data documented in the [NOTICE](NOTICE) file.

## Acknowledgements

CroCoDiLight builds on [CroCo](https://github.com/naver/croco) (Weinzaepfel et al.),
licensed under CC BY-NC-SA 4.0 by Naver Corporation.

Model training was performed on the Viking cluster, a high performance compute facility
provided by the University of York. We are grateful for computational support from the
University of York, IT Services and the Research IT team.

## AI Usage

[Claude Code](https://claude.ai/download) (Anthropic) was used as a development tool to
package the project as an installable Python module (pyproject.toml), refactor inference
scripts (removing hard-coded paths and extracting shared model loading and utility
functions), and build the Gradio demos. All core research code, model architectures, and
training procedures were designed and written by the authors.

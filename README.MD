# CroCoDiLight

CroCo-based Disentangled Lighting for image delighting and relighting. Extends CroCo (Cross-view Completion) to separate
content from illumination in latent space, enabling shadow removal, albedo estimation, lighting transfer, and
interpolation.

## Setup

1. Create a conda environment and install dependencies:

```bash
conda create -n croco python=3.10 -y
conda activate croco
conda install pytorch torchvision -c pytorch
pip install -e .
```

2. (Optional) Compile CUDA kernels for RoPE positional embeddings. A pure Python fallback is used automatically if these
   are not compiled. It is possible to install `cuda-toolkit` instead of `cuda-cudart-dev` and `cuda-nvcc` if you wanted
   a complete CUDA installation. The system CUDA installation should also work but has not been tested.

```bash
conda install cuda-cudart-dev cuda-nvcc ninja -c nvidia
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
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

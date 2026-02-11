# CroCoDiLight

## Environment Setup

1. Create a conda environment and install dependencies:

```bash
conda create -n croco python=3.10 -y
conda activate croco
conda install pytorch torchvision -c pytorch
pip install -e .
```

2. (Optional) Compile CUDA kernels for RoPE positional embeddings. This provides GPU acceleration but is not required --
   a pure Python fallback will be used automatically if the compiled version is not available.

First, install the CUDA development headers and build tools into the conda environment. You can also install the full
`cuda-toolkit` package instead of `cuda-cudard-dev` and `cuda-nvcc`.

```bash
conda install cuda-cudart-dev cuda-nvcc ninja -c nvidia 
```

Then compile the CUDA RoPE positional embeddings:

```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

3. Download the CroCo v2 pre-trained weights:

```bash
mkdir -p pretrained_models/
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth -P pretrained_models/
```

4. Log in to Weights & Biases (used for training visualisation):

```bash
wandb login
```

## Training

There are three main stages to training, one of them repeated.

### Step 1: Pretrain the single-view decoder

```bash
python scripts/pretrain_relight_decoder.py
```

### Step 2: Train the relighting model

Train the base model with the Delighting and Relighting transformers:

```bash
python scripts/train_relight_model.py
```

### Step 3: Train task-specific mappers

Edit the function call at the end of `scripts/train_lighting_mapper.py` to be either `"shadow"` or `"albedo"` depending
on which latent transformation model you want to train. Then run:

```bash
python scripts/train_lighting_mapper.py
```

## Evaluation

### Albedo estimation (IIW)

```bash
python scripts/iiw.py
python scripts/whdr_full_evaluation.py
```

The first command estimates albedo on the IIW dataset and calculates WHDR across the whole dataset. The second takes the
precomputed results and calculates WHDR on the test set.

### Shadow removal

```bash
python scripts/shadow_removal.py
python scripts/shadow_eval.py "data/sr_outputs"
```

The first runs shadow removal on all the specified datasets, and the second evaluates all of the metrics.

## Inference

`scripts/interpolate_lighting.py` and `scripts/swap_lighting.py` demonstrate ways of working with the model for
relighting tasks.

## Pretrained Models

The following model files are expected in `pretrained_models/`:

| File                                | Description                                                                             |
|-------------------------------------|-----------------------------------------------------------------------------------------|
| `CroCo_V2_ViTLarge_BaseDecoder.pth` | Base CroCo v2 model (download above)                                                    |
| `croco_decoder_pretrained.pth`      | Pretrained decoder (output of step 1)                                                   |
| `croco_relight_full.pth`            | Full relighting model with CroCo encoder kwargs (output of step 2)                      |
| `{task}_mapper.pth`                 | Task-specific mappers, e.g. `shadow_mapper.pth`, `albedo_mapper.pth` (output of step 3) |

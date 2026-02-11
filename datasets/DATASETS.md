# Datasets

Datasets used in CroCoDiLight. Place each dataset under the `datasets/` directory.

## Shadow removal

| Dataset | Download | Expected path | Used in |
|---------|----------|---------------|---------|
| SRD | [Qu et al.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qu_DeshadowNet_A_Multi-Context_CVPR_2017_paper.pdf) ([download via DC-ShadowNet](https://github.com/jinyeying/DC-ShadowNet-Hard-and-Soft-Shadow-Removal)) | `datasets/SRD/` | Evaluation, Training (Steps 2 & 3) |
| ISTD+ | [ST-CGAN (Wang et al.)](https://github.com/DeepInsight-PCALab/ST-CGAN) + [adjusted GT (Le & Samaras)](https://github.com/cvlab-stonybrook/SID) | `datasets/ISTD+/` | Evaluation, Training (Steps 2 & 3) |
| WSRD+ | [WSRD-DNSR (Vasluianu et al.)](https://github.com/fvasluianu97/WSRD-DNSR) | `datasets/WSRD+/` | Visual evaluation, Training (Steps 2 & 3) |
| INS | [OmniSR (Xu et al.)](https://blackjoke76.github.io/Projects/OmniSR/) | `datasets/INS/` | Ablation study |

Expected directory layout:

```
datasets/SRD/         train/ and test/ with shadow/ and shadow_free/ subfolders
datasets/ISTD+/       train_A/, train_C_fixed_ours/, test_A/, test_C_fixed_official/ (ISTD+ fixed ground truths)
datasets/WSRD+/       train/ and val/ with input/ and gt/ subfolders
datasets/INS/         test/ with origin/ and shadow_free/ subfolders
```

## Intrinsic image decomposition

| Dataset | Download | Expected path | Used in |
|---------|----------|---------------|---------|
| IIW | [Intrinsic Images in the Wild (Bell et al.)](http://opensurfaces.cs.cornell.edu/publications/intrinsic/) | `datasets/IIW/` | Evaluation |
| ML-HyperSim | [Apple ML-HyperSim](https://github.com/apple/ml-hypersim) | `datasets/ML-HyperSim/` | Training (Steps 2 & 3) (first frame per trajectory) |
| CGIntrinsics | [CGIntrinsics (Li & Snavely)](https://www.cs.cornell.edu/projects/cgintrinsics/) | `datasets/CGIntrinsics/` | Training (Steps 2 & 3) |

Each IIW image has a paired `.json` judgements file used for WHDR computation.

## Timelapse / multi-illumination

| Dataset | Download | Expected path | Used in |
|---------|----------|---------------|---------|
| BigTime | [BigTime (Li & Snavely)](https://www.cs.cornell.edu/projects/bigtime/) | `datasets/BigTime/` | Training (Step 2) |
| MIT Multi-Illumination | [MIT (Murmann et al.)](https://projects.csail.mit.edu/illumination/) | `datasets/Multi_Illumination/` | Training (Step 2) |

BigTime: extract the downloaded zip into `datasets/BigTime/`. The data is nested at
`BigTime/phoenix/S6/zl548/AMOS/BigTime_v1/` and referenced as such by the training scripts.

MIT Multi-Illumination: the main dataset and the test split are separate downloads. Place the
training images under `datasets/Multi_Illumination/train/` and the test split under
`datasets/Multi_Illumination/test/`.

## Pretraining

| Dataset | Download | Expected path | Used in |
|---------|----------|---------------|---------|
| ImageNet (ILSVRC 2012) | [PyTorch ImageNet](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html) | `datasets/ImageNet/` | Training (Step 1) |

Standard ImageNet layout with `train/` and `val/` splits, as expected by `torchvision.datasets.ImageNet`.

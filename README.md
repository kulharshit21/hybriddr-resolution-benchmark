# HybridDR: resolution, fusion and diabetic-retinopathy grading

Public research implementation for **The Mild-Grade Challenge: Resolution and Fusion in Cross-Dataset Diabetic Retinopathy Grading**.

This repository contains code, experiment configurations, synthetic unit tests and usage documentation. It contains **no experimental results, patient-level predictions, retinal photographs, datasets, caches, model weights or checkpoints**. It is a separate repository with a fresh history. The models are research prototypes, not clinically validated or regulated diagnostic products.

## Repository map

| Directory | Purpose |
|---|---|
| `pipeline/` | Dataset preparation, quality/duplicate audits, original architecture and loss controls, training, calibration, evaluation and inference |
| `experiments/v2/` | Residual-scaled fusion and loss-blend development code |
| `experiments/v3-targeted/` | Targeted loss comparisons and grade-specific diagnostics |
| `experiments/v4-classifier/` | Frozen-feature classifier adaptation and prediction diagnostics |
| `experiments/v5-resolution/` | Initial resolution-screen implementation |
| `experiments/v6/` | Matched-resolution aligned-fusion training and evaluation |
| `experiments/v7/` | Matched CNN/gated controls at the higher resolution |
| `analysis/` | Paired prediction review, examination-bootstrap uncertainty and qualitative attribution |
| `audits/` | MESSIDOR exclusion, label-source and public-pairing checks |
| `environment/` | Original dependency pins and separately captured package inventory |

## Environment

Use an isolated environment. The original requirements pin torch 2.11.0 / torchvision 0.26.0; a later DGX inventory recorded torch 2.15.0 / torchvision 0.29.2. Both recorded timm 1.0.29. These records are preserved separately: they are **not interchangeable lockfiles**, and an upgrade chronology was not recovered. Select a compatible PyTorch/torchvision pair for your platform using [the official installer](https://pytorch.org/get-started/locally/). Do not infer bitwise reproduction from a successful installation. Additional analysis packages are listed separately.

Core dependencies include PyTorch, torchvision, timm, NumPy, pandas, SciPy, scikit-learn, Pillow and psutil. Auditing additionally uses scikit-image, OpenCV and Kornia. CPU-only checks use synthetic examples and do not require benchmark data or pretrained-weight downloads.

```bash
python -m venv .venv
# Activate the environment, install a compatible torch/torchvision pair, then the remaining requirements.
python pipeline/run.py --help
python pipeline/predict.py --help
cd pipeline
python -m unittest test_pipeline.PipelineTests.test_all_recipes_forward_backward test_pipeline.PipelineTests.test_attention_query_key_gradients test_pipeline.PipelineTests.test_metrics_perfect
```

## Obtain data and pretrained encoders separately

- [EyePACS competition](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)
- [APTOS 2019 competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
- [Original MESSIDOR-2 provider and terms](https://www.adcis.net/en/third-party/messidor2/)
- [Google Brain MESSIDOR-2 adjudicated labels](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades)
- [Public examination-pairing mirror](https://github.com/ok1zjf/RFM/blob/b8c9322863b5e2d79df964e524d74ad53d1d1b7a/datasets/messidor2_pairs.csv)
- [DINOv2](https://github.com/facebookresearch/dinov2), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [timm](https://github.com/huggingface/pytorch-image-models)
- [LightGlue](https://github.com/cvg/LightGlue), [ALIKED](https://github.com/Shiaoming/ALIKED), [SSCD](https://github.com/facebookresearch/sscd-copy-detection)

The image collection, adjudicated label release and public pairing mirror are distinct sources. The mirror does not establish original-spreadsheet authentication or unique patient identities across examinations. A historical acquisition lead was [this Kaggle image mirror](https://www.kaggle.com/datasets/nuruzzamankallol/messidor-2-rebuild); the completed image-download version/archive identity was not recovered, so it is **not asserted as authenticated acquisition provenance**. Use the original provider under applicable terms. Do not upload data or weights to this repository. Upstream assets retain their own terms and attribution requirements.

## Reproduction boundaries and execution

Configure your dataset root, output root and allocated GPU IDs explicitly. `pipeline/run.py` exposes prepare/train/evaluate/report stages. Non-smoke training requires the data-review binding described in `docs/REPRODUCIBILITY.md`. A fresh prepared manifest will have its own identity. Archived V5–V7 launchers deliberately bind to the study manifest and registry; they fail on a different manifest instead of silently claiming the same experiment. Inspect/adapt those bindings explicitly for a new study and use a fresh output directory. Keep the archived model and loss implementations separate when comparing stages.

The whole-image classifier path uses EXIF-oriented RGB images cached as square Lanczos resamplings, followed by model-input resizing and channel standardization. Foreground cropping belongs to duplicate auditing, not classifier preprocessing. Checkpoints are selected using the declared validation rule; evaluation must not tune on test outputs. Examination bootstrap samples examinations conditional on fixed trained models. The attribution implementation is a first-gradient-power Grad-CAM-style approximation, not exact higher-derivative Grad-CAM++ or validated lesion localization.

Site-specific MAMAGuard stop scripts, migration helpers, cloud backup/upload wrappers, credentials and private absolute paths are excluded. Public launchers do not grant permission to occupy shared GPUs: use only resources allocated to you. A few site-operation tests/default paths were removed or made portable; model/loss mathematics and experiment configuration values were not changed. See `PUBLIC_RELEASE_NOTES.md`.

## Citation and licensing

See `CITATION.md` for the manuscript author/title metadata. No publication DOI or acceptance is claimed. No software licence has been selected for this release; public visibility is not an additional licence grant. Dataset and pretrained-model rights are independent of repository visibility.

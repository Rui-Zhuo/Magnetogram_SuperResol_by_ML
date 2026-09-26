# HMI to Hinode/SP magnetogram reconstruction

Code and pretrained models for **Full-Disk Solar Magnetogram Reconstruction to Hinode/SP Resolution Using an Improved Arbitrary-Scale Super-Resolution Network**.

The repository includes data preparation, training, inference, evaluation, and existing test examples. Model names follow Table 2 of the manuscript; **PM-LTEW-CC** is the primary model.

## Quick start

Requires Python 3.10 or newer. Run from the repository root:

```bash
git clone https://github.com/Rui-Zhuo/Magnetogram_SuperResol_by_ML.git
cd Magnetogram_SuperResol_by_ML
python -m pip install -e .
python -m magnetosr.inference --input examples/data/20121027_034505.npz --output outputs/predictions
python -m magnetosr.evaluate --predictions outputs/predictions --data examples/data --output outputs/evaluation
```

Inference defaults to `checkpoints/pm_ltew_cc.pth` on CPU. Use `--device cuda` for a CUDA-enabled PyTorch installation.

## Models

| Model (Table 2) | Pretrained weights | Training configuration |
|---|---|---|
| **PM-LTEW-CC** | [pm_ltew_cc.pth](checkpoints/pm_ltew_cc.pth) | [train_pm_ltew_cc.yaml](configs/train_pm_ltew_cc.yaml) |
| PM-LTEW | [pm_ltew.pth](checkpoints/pm_ltew.pth) | [train_pm_ltew.yaml](configs/train_pm_ltew.yaml) |
| LTEW-CC | [ltew_cc.pth](checkpoints/ltew_cc.pth) | [train_ltew_cc.yaml](configs/train_ltew_cc.yaml) |
| LTEW | [ltew.pth](checkpoints/ltew.pth) | [train_ltew.yaml](configs/train_ltew.yaml) |
| PM-RCAN-CC | [pm_rcan_cc.pth](checkpoints/pm_rcan_cc.pth) | [train_pm_rcan_cc.yaml](configs/train_pm_rcan_cc.yaml) |
| PM-SRCNN-CC | [pm_srcnn_cc.pth](checkpoints/pm_srcnn_cc.pth) | [train_pm_srcnn_cc.yaml](configs/train_pm_srcnn_cc.yaml) |

PM denotes position modulation; CC denotes the correlation-coefficient loss term. The shared LTEW implementation is `magnetosr/models/lte_warp.py`, registered as `lte-warp`. Run dates, epochs and checksums are in [the checkpoint manifest](checkpoints/manifest.json); original run configurations are in [`configs/historical/`](configs/historical).

## Data preparation and training

The workflow downloads HMI/SP observations, performs alignment and local registration, crops paired patches, and writes NPZ files containing `HMIfield`, `SPfield` and `Txy`. Input and target shapes are 200×200 and 312×336. Magnetic fields are in G; `Txy` is in arcsec.

See [commands and directory layout](docs/usage.md), [data sources and preparation](docs/data_access.md), and [metric definitions](docs/metrics.md).

```bash
python -m magnetosr.train --config configs/train_pm_ltew_cc.yaml --data data/paired --output outputs/train_pm_ltew_cc --device cuda
```

## Test examples

Existing test result for `20121027_034505`; its paired data and saved prediction are included under `examples/`.

![Test sample 20121027](examples/figures/20121027_034505.png)

## Data availability and attribution

This release includes example data and the train/validation/test split. The complete paired dataset and full-disk products will be archived separately.

Based on [LTEW](https://github.com/jaewon-lee-b/ltew) and the [SOT/SP pointing updates](https://github.com/dfouhey/SOTSPPointing). Please cite the accompanying manuscript and the upstream methods and data sources. See [licenses and attribution](third_party/README.md).

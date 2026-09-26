# Commands and directory layout

## Layout

```text
models/              # LTE-warp, PM-RCAN-CC, PM-SRCNN-CC and shared layers
pre-trained/         # six checkpoints and checksum manifest
configs/             # train_*.yaml, test_*.yaml and historical configurations
examples/            # paired test data, saved predictions and figures
full-disk/           # three full-disk cases, original inputs, results and figures
magnetosr/           # Python processing, training, inference and plotting tools
data/                # dataset split and test metadata
docs/                # detailed workflow and metric definitions
```

## Prepare data

Install the solar-data dependencies and obtain the pointing FITS archive described in [data access](data_access.md). Place the selected pointing files directly in `data/raw/pointing/`, named by observation ID.

```bash
python -m pip install -e ".[solar]"
python -m magnetosr.download sp --level 2 --output data/raw/sp_l2
python -m magnetosr.download sp --level 2.1 --output data/raw/sp_l21
python -m magnetosr.download hmi --email YOUR_REGISTERED_JSOC_EMAIL --output data/raw/hmi
python -m magnetosr.coalign --pointing data/raw/pointing --sp-l2 data/raw/sp_l2 --sp-l21 data/raw/sp_l21 --hmi data/raw/hmi --pairs data/raw/hmi/pairs.csv --output data/aligned
python -m magnetosr.prepare --input data/aligned --output data/paired --split data/splits/dataset_split_10115.json
```

The supplied split has 6,069 training, 2,023 validation and 2,023 test samples. Use the same split for every experiment. For a new dataset only:

```bash
python -m magnetosr.split --data data/paired --output data/splits/new_split.json --seed 42
```

## Train, predict and evaluate

```bash
python -m magnetosr.train --config configs/train_pm_ltew_cc.yaml --data data/paired --output outputs/train_pm_ltew_cc --device cuda
python -m magnetosr.inference --config configs/test_pm_ltew_cc.yaml
python -m magnetosr.evaluate --predictions outputs/pm_ltew_cc --data examples/data --output outputs/evaluation
python -m magnetosr.plot_samples --data examples/data --predictions outputs/pm_ltew_cc --output outputs/figures
```

Select other experiments by replacing the configuration/checkpoint with the corresponding Table 2 name from the README. `--chunk-size` controls inference memory. Training checkpoints written as `last.pth` support `--resume`; distributed inference weights omit optimizer state. The input order is `[HMIfield, Txy]`; both channels and the target are divided by 200, and inference outputs are converted back to G.

## Model settings

| Table 2 name | File stem | Position modulation | CC weight |
|---|---|---|---|
| PM-LTEW-CC | `pm_ltew_cc` | Yes | 0.01 |
| PM-LTEW | `pm_ltew` | Yes | 0 |
| LTEW-CC | `ltew_cc` | No | 0.01 |
| LTEW | `ltew` | No | 0 |
| PM-RCAN-CC | `pm_rcan_cc` | Yes | 0.01 |
| PM-SRCNN-CC | `pm_srcnn_cc` | Yes | 0.01 |

Bicubic, Empirical-D and Empirical-Z are the non-learned baselines in Table 2 and have no training checkpoints. The shared evaluator includes the bicubic baseline. More implementation details are in [implementation notes](reproducibility.md).

## Full-disk results and magnetogram plots

The [full-disk guide](../full-disk/README.md) explains the three included applications and lossless reconstruction-array storage. To display a downloaded HMI FITS map or a paired NPZ:

```bash
python -m magnetosr.plot_magnetogram --input full-disk/solar-maximum/hmi.fits --hdu 1 --output outputs/hmi.png
python -m magnetosr.plot_magnetogram --input examples/data/20121027_034505.npz --key SPfield --output outputs/sp-patch.png
```

FITS plotting requires the solar dependencies. These plots use array coordinates; the original paper composites are included under each full-disk case.

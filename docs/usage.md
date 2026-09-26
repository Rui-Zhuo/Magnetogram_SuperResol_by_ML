# Commands and directory layout

## Layout

```text
magnetosr/
  models/
    lte_warp.py       # shared LTEW architecture
    pm_rcan_cc.py     # PM-RCAN-CC
    pm_srcnn_cc.py    # PM-SRCNN-CC
  download.py        # HMI and SP downloads
  coalign.py         # affine/pointing alignment
  prepare.py         # local registration, crop and NPZ generation
  train.py           # shared training entry
  inference.py       # CPU/CUDA inference
  evaluate.py        # metrics and grouped summaries
configs/
  train_*.yaml       # runnable configurations named after Table 2 models
  historical/       # original run configurations, using the same model names
checkpoints/         # weights and provenance manifest
examples/            # paired test data, existing predictions and figures
data/                # split and test metadata
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
python -m magnetosr.inference --checkpoint checkpoints/pm_ltew_cc.pth --input examples/data --output outputs/pm_ltew_cc --device cpu
python -m magnetosr.evaluate --predictions outputs/pm_ltew_cc --data examples/data --output outputs/evaluation
python -m magnetosr.plot_samples --data examples/data --predictions outputs/pm_ltew_cc --output outputs/figures
```

Select other experiments by replacing the configuration/checkpoint with the corresponding Table 2 name from the README. `--chunk-size` controls inference memory. Training checkpoints written as `last.pth` support `--resume`; distributed inference weights omit optimizer state. The input order is `[HMIfield, Txy]`; both channels and the target are divided by 200, and inference outputs are converted back to G.

## Model names and original runs

| Table 2 name | File stem | Original run | Position modulation | CC weight |
|---|---|---|---|---|
| PM-LTEW-CC | `pm_ltew_cc` | 20260416 | Yes | 0.01 |
| PM-LTEW | `pm_ltew` | 20260831 | Yes | 0 |
| LTEW-CC | `ltew_cc` | 20260901 | No | 0.01 |
| LTEW | `ltew` | 20260902 | No | 0 |
| PM-RCAN-CC | `pm_rcan_cc` | 20260912 | Yes | 0.01 |
| PM-SRCNN-CC | `pm_srcnn_cc` | 20260915 | Yes | 0.01 |

Bicubic, Empirical-D and Empirical-Z are the non-learned baselines in Table 2 and have no training checkpoints. The shared evaluator includes the bicubic baseline. More implementation details are in [implementation notes](reproducibility.md).

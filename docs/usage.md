# Commands and directory layout

| Folder | Purpose |
|---|---|
| `prep-dataset/` | HMI/SP download, alignment, registration and paired NPZ preparation |
| `models/` | LTE-warp, PM-RCAN-CC, PM-SRCNN-CC and shared layers |
| `pre-trained/` | Six pretrained model files and checksums |
| `configs/` | Named training/test configurations and original configuration references |
| `training/` · `inference/` | Training and patch inference |
| `evaluation/` | One metric implementation, shared evaluation and plotting |
| `full-disk/` | Full-disk download, reconstruction and plotting code |
| `examples/application/` | Three full-disk paper previews and case metadata |
| `examples/data/` · `examples/predictions/` | Paired test examples and saved predictions |
| `data/` · `docs/` | Dataset split, metadata and detailed documentation |

Install with `python -m pip install -e ".[solar]"` from the repository root. The hyphenated folders are installed as Python packages named `prep_dataset` and `full_disk`.

## Prepare data

Follow the [dataset preparation guide](../prep-dataset/README.md). The included split has 6,069 train, 2,023 validation and 2,023 test samples. For a new dataset only:

```bash
python -m prep_dataset.split --data data/paired --output data/splits/new_split.json --seed 42
```

## Train, predict and evaluate

```bash
python -m training.train --config configs/train_pm_ltew_cc.yaml --data data/paired --output outputs/training --device cuda
python -m inference.predict --config configs/test_pm_ltew_cc.yaml
python -m evaluation.evaluate --predictions outputs/pm_ltew_cc --data examples/data --output outputs/evaluation
python -m evaluation.plot_samples --data examples/data --predictions outputs/pm_ltew_cc --output outputs/figures
```

Select other Table 2 models using the matching [configuration](../configs/README.md). Explicit inference options override the test YAML. Training `last.pth` files support `--resume`; distributed inference checkpoints omit optimizer state. Inputs are `[HMIfield, Txy]`; both channels and training targets are divided by 200. Prediction arrays are converted back to G.

## Full-disk applications

The [full-disk guide](../full-disk/README.md) provides the complete download, patch prediction, assembly, filling and plotting sequence. The [case gallery](../examples/application/README.md) contains the paper examples without large result arrays.

## Display a magnetic-field array

```bash
python -m evaluation.plot_magnetogram --input PATH_TO_HMI.fits --hdu 1 --output outputs/hmi.png
python -m evaluation.plot_magnetogram --input examples/data/20121027_034505.npz --key SPfield --output outputs/sp-patch.png
```

FITS plotting requires the solar dependencies. Plots use array coordinates; the original manuscript composites are provided in the case gallery.

# Model configurations

Names follow Table 2 of the manuscript.

| Model | Training | Test/inference | Weights |
|---|---|---|---|
| **PM-LTEW-CC** | [train_pm_ltew_cc.yaml](train_pm_ltew_cc.yaml) | [test_pm_ltew_cc.yaml](test_pm_ltew_cc.yaml) | [pm_ltew_cc.pth](../pre-trained/pm_ltew_cc.pth) |
| PM-LTEW | [train_pm_ltew.yaml](train_pm_ltew.yaml) | [test_pm_ltew.yaml](test_pm_ltew.yaml) | [pm_ltew.pth](../pre-trained/pm_ltew.pth) |
| LTEW-CC | [train_ltew_cc.yaml](train_ltew_cc.yaml) | [test_ltew_cc.yaml](test_ltew_cc.yaml) | [ltew_cc.pth](../pre-trained/ltew_cc.pth) |
| LTEW | [train_ltew.yaml](train_ltew.yaml) | [test_ltew.yaml](test_ltew.yaml) | [ltew.pth](../pre-trained/ltew.pth) |
| PM-RCAN-CC | [train_pm_rcan_cc.yaml](train_pm_rcan_cc.yaml) | [test_pm_rcan_cc.yaml](test_pm_rcan_cc.yaml) | [pm_rcan_cc.pth](../pre-trained/pm_rcan_cc.pth) |
| PM-SRCNN-CC | [train_pm_srcnn_cc.yaml](train_pm_srcnn_cc.yaml) | [test_pm_srcnn_cc.yaml](test_pm_srcnn_cc.yaml) | [pm_srcnn_cc.pth](../pre-trained/pm_srcnn_cc.pth) |

Use `python -m training.train --config configs/train_pm_ltew_cc.yaml --data data/paired --output outputs/training` for training and `python -m inference.predict --config configs/test_pm_ltew_cc.yaml` for inference. Paths are relative to the repository root. Explicit command-line inference options override YAML values.

The four LTEW variants share `models/lte_warp.py`, registered as `lte-warp`; `use_pm` and `cc_weight` distinguish the experiments. PM-RCAN-CC and PM-SRCNN-CC have separate architecture files. Bicubic, Empirical-D and Empirical-Z are non-learned baselines without checkpoints.

The four LTEW training YAMLs explicitly select `position_mode: patch_center`, restore the normalized radius by 200, use a 960-arcsec reference radius, and clip the modulation to [0.5, 1.5]. The test YAMLs point to unchanged historical checkpoints; loading an unversioned LTEW checkpoint warns and uses the explicit `legacy_query_min` compatibility path. Add `--position-mode patch_center` to opt into a different PM rule with those tensors, with a warning that predictions change. These new training recipes and the retained checkpoints are therefore distinguished explicitly; see [checkpoint compatibility](../pre-trained/README.md).

[`historical/`](historical) contains sanitised copies of the original training configurations for reference. The top-level `train_*.yaml` files are for the released training entry point.

All six released `train_*.yaml` files specify 300 epochs, matching the author-confirmed training schedule for the manuscript experiments. The `epoch_max: 200` entry retained in `historical/pm_srcnn_cc.yaml` is a stale historical setting and should not be used to infer the duration of the paper experiment. The epoch stored in each distributed checkpoint is the selected checkpoint's epoch, rather than the total training duration.

# Evaluation and plotting

[`metrics.py`](metrics.py) is the single implementation of the paper's evaluation metrics. All model comparisons and test-sample plots use it. See [definitions and aggregation conventions](../docs/metrics.md).

```bash
python -m evaluation.evaluate --predictions outputs/pm_ltew_cc --data examples/data --output outputs/evaluation
python -m evaluation.plot_samples --data examples/data --predictions outputs/pm_ltew_cc --output outputs/figures
python -m evaluation.plot_magnetogram --input examples/data/20121027_034505.npz --key SPfield --output outputs/sp-patch.png
python -m evaluation.plot_magnetogram --input PATH_TO_HMI.fits --hdu 1 --output outputs/hmi.png
```

The evaluator accepts any model's prediction directory, using the `pred` array and matching filenames. Its shared grouping logic uses the supplied test metadata. `plot_magnetogram.py` displays one FITS or NPZ array in pixel coordinates; HMI uses HDU 1 and SP L2.1 uses HDU 4.

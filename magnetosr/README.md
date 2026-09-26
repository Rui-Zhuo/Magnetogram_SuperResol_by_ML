# Processing and command-line tools

`magnetosr` is the Python package name used by commands such as `python -m magnetosr.inference`. It contains the workflow tools; model architecture definitions are in [`models/`](../models).

| Module | Purpose |
|---|---|
| `download.py` | Download HMI and SP observations |
| `coalign.py` · `registration.py` | Coarse alignment and local registration |
| `prepare.py` · `split.py` | Crop paired patches, generate NPZ files and dataset splits |
| `train.py` · `inference.py` | Train models and reconstruct magnetograms |
| `metrics.py` · `evaluate.py` | Metric definitions and shared evaluation |
| `plot_samples.py` | Plot paired test results |
| `plot_magnetogram.py` | Display downloaded FITS maps, NPZ arrays or full-disk results |
| `full_disk.py` | Read and reassemble the supplied full-disk result files |
| `coordinates.py` | Coordinate-grid utilities used by models and inference |

See [usage](../docs/usage.md) for commands.

# Model architectures

- `lte_warp.py`: portable Fourier decoder shared by PM-LTEW-CC, PM-LTEW, LTEW-CC and LTEW; new models use patch-center PM, while unversioned checkpoints load through an explicit legacy-PM path.
- `pm_rcan_cc.py`: PM-RCAN-CC.
- `pm_srcnn_cc.py`: PM-SRCNN-CC.
- `rcan.py` and `mlp.py`: shared encoder and prediction layers.
- `models.py` and `__init__.py`: model registry and factory.

Names and experiment settings follow manuscript Table 2; see [configs](../configs/README.md). See [usage](../docs/usage.md) for training, inference and data processing commands.

The portable phase layer retains two cell-size features from the supplied checkpoints. No ten-feature/Jacobian weight conversion is performed; see [implementation details](../docs/reproducibility.md).

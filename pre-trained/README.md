# Pretrained models

The six weights use the model names from manuscript Table 2. **PM-LTEW-CC** is the primary model. See [configuration and weight links](../configs/README.md).

These inference checkpoints retain model specifications and trained tensors, without optimizer state. The [manifest](manifest.json) lists each model, selected epoch, architecture, position-modulation setting, CC weight, file size and SHA-256 checksum. `source_sha256` identifies the original research checkpoint; `sha256` identifies the distributed file.

The four LTEW checkpoint specifications have no recorded PM-version field and retain a two-cell phase layer. The loader warns and selects the explicit `legacy_query_min` path to preserve their earlier public-release behavior. This differs from the patch-center PM now used by new models and training configurations. `--position-mode patch_center` is an explicit, warned inference override that changes predictions; it is not a verified reproduction of the saved paper results. Newly trained checkpoints store their PM mode. The two fixed-grid RCAN/SRCNN architectures already use patch-center PM. No distributed weight tensors have been modified, reshaped or padded; see [implementation and compatibility details](../docs/reproducibility.md).

```bash
python -m inference.predict --config configs/test_pm_ltew_cc.yaml
```

# Pretrained models

The six weights use the model names from manuscript Table 2. **PM-LTEW-CC** is the primary model. See [configuration and weight links](../configs/README.md).

These inference checkpoints retain model specifications and trained tensors, without optimizer state. The [manifest](manifest.json) lists each model, selected epoch, architecture, position-modulation setting, CC weight, file size and SHA-256 checksum. `source_sha256` identifies the original research checkpoint; `sha256` identifies the distributed file.

```bash
python -m inference.predict --config configs/test_pm_ltew_cc.yaml
```

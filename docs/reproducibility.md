# Implementation and provenance

## Primary model

The distributed primary-model checkpoint is **PM-LTEW-CC**, selected at epoch 115. Its tensor values are retained; optimizer state is omitted from the public inference checkpoint. `pre-trained/manifest.json` identifies original and exported files with SHA-256 hashes and records compatibility settings. The five other checkpoints are PM-LTEW, LTEW-CC, LTEW, PM-RCAN-CC and PM-SRCNN-CC. A selected checkpoint epoch is distinct from the author-confirmed 300-epoch training schedule.

The portable Fourier decoder is registered as `lte-warp`, with package-relative imports, CPU/CUDA handling and an explicit no-PM switch. It retains the distributed weights' two-dimensional cell-size phase layer. It is not a port of the research source's ten-feature Jacobian decoder; no two-feature checkpoint is converted into a ten-feature checkpoint. RCAN/MLP parameter names and the supplied checkpoint tensors are preserved.

## Position modulation

New model construction and all four released LTEW training configurations use `position_mode: patch_center`. HMI and projected radius are divided by 200 at input. The normalized patch-center radius is converted back to arcsec, then divided by the fixed reference solar radius of 960 arcsec:

```python
center_radius_arcsec = inp[:, 1, H // 2, W // 2] * 200
dist_norm = center_radius_arcsec / 960
modulation = (1 - gamma * dist_norm + beta).clamp(0.5, 1.5)
output = implicit_prediction + bilinear_hmi * modulation
```

`gamma` and `beta` are trainable parameters. One scalar is applied to all query locations in each patch, independently for each batch member. No-PM ablations add unmodulated bilinear HMI instead. Query chunking cannot change the center-radius weight. Fixed-grid RCAN/SRCNN use the same center-radius normalization and clipping.

### Historical checkpoint compatibility

The distributed LTEW checkpoint specifications predate explicit PM-version metadata. `models.make(..., load_sd=True)` and `inference.predict.load_model` select `legacy_query_min` for such specifications and warn the user. This path preserves the earlier public-release formula, `(interpolated_normalized_radius - minimum_query_radius) / 960`; it does not restore the radius by 200. Chunked legacy inference calculates the full-query minimum once. This is a compatibility rule for the existing assets, not a claim that their original training PM provenance has been independently established.

An explicit `--position-mode patch_center` override loads the same tensors with the center-radius formula and warns that predictions change. No historical-reproduction claim is made for that override. New training saves `position_mode` in the checkpoint model arguments, even if a custom training YAML omitted it. Resuming an old checkpoint under a different PM mode is rejected. The tensor dimensions and bytes of all six distributed weight files remain unchanged.

## Processing and training

1. Download SP L2/L2.1 and time-matched HMI observations using explicit sample IDs.
2. Use published pointing/affine FITS metadata, slit positions and HMI WCS to form coarse pairs.
3. Resample SP onto the HMI grid, apply local phase-correlation registration, centre-crop to 200x200 and resample the target to 312x336.
4. Save `HMIfield`, `SPfield` and `Txy` in NPZ files, preserving Gauss and arcsec units.
5. Use the existing train/validation/test split for all models. Supply HMI and radius as two normalized channels.
6. Train with AdamW and `(1-w)*L1 + w*(1-CC)`. Save validation-best weights. Training and validation never draw from the test list.
7. Predict on requested output coordinates, convert back to Gauss, and evaluate against the paired NPZ reference.

Local-registration kernels are retained. Removed SciPy `interp2d` calls are replaced with regular-grid `RectBivariateSpline`. Public drivers expose paths/options instead of embedding machine directories or cluster accounts. Historical configurations are sanitised copies; the simplified runner supplies common configurations for new experiments. Training is not rerun during release preparation.

## Existing examples and figures

Examples use the supplied paired NPZ data, saved predictions and existing figure PNGs. `examples/manifest.json` maps observation IDs to original prediction filenames. Public prediction NPZ files contain `HMIfield`, `Txy`, `SPfield` from paired data and saved `pred` reshaped to the target grid. Predictions are not recalculated to replace existing research results. The evaluator reads its reference from the paired-data directory rather than relying on the historical output-file target normalization.

The three full-disk paper figures and observing information are in `examples/application`. Large existing arrays are omitted from the current tree. `full-disk` provides download, centred patch inference, tile assembly, cubic filling and plotting commands, adapted from the retained research scripts. Newly generated files include a binary `quality_flag` (1: finite model output; 0: fallback, missing or invalid) and the more detailed `source_mask` (0: invalid; 1: model; 2: interpolation). Neither mask quantifies uncertainty or retrospectively supplies provenance for historical files.

Release checks cover file completeness, syntax, entry-point availability, split membership and model loading. PM tests additionally cover absolute-radius scaling, uniform per-patch weights, batch independence, clipping, no-PM behavior, chunk invariance, versioned checkpoint loading and resume safeguards. Agreement between current code, historical results and paper figures is outside this release-preparation scope. A complete data archive remains a separate task.

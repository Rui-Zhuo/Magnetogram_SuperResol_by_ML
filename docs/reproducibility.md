# Implementation and provenance

## Primary model

The primary model is **PM-LTEW-CC**, epoch 115. Its model specification and tensor values are retained; optimizer state is omitted from the public inference checkpoint. `pre-trained/manifest.json` identifies original and exported files with SHA-256 hashes. The five ablations are PM-LTEW, LTEW-CC, LTEW, PM-RCAN-CC and PM-SRCNN-CC.

The LTE-warp implementation comes from retained original source, with package-relative imports, CPU/CUDA device handling and an explicit no-position-modulation switch. RCAN/MLP parameter names are preserved. The paper's PM-LTEW-CC model is registered as `lte-warp`; its phase layer receives a two-dimensional cell-size vector.

## Position modulation

The LTE-warp path divides both HMI and projected radius by 200. For interpolated normalized radius `d`, the skip connection is:

```python
dist_norm = (d - d.min()) / 960
modulation = (1 - gamma * dist_norm + beta).clamp(0.5, 1.5)
output = implicit_prediction + bilinear_hmi * modulation
```

`gamma` and `beta` are trained parameters. No-PM ablations add unmodulated bilinear HMI instead. The radius minimum spans the complete query tensor. Chunked inference computes the minimum once and reuses it across chunks. Each inference call processes one observation.

Fixed-grid RCAN/SRCNN implementations use the input patch's centre radius, scale it back by 200/960, and apply their learned, clamped affine modulation. Their target grid is fixed at 312x336. These details are retained in their respective model definitions.

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

The three full-disk paper figures and observing information are in `examples/application`. Large existing arrays are omitted from the current tree. `full-disk` provides download, centred patch inference, tile assembly, cubic filling and plotting commands, adapted from the retained research scripts. Newly generated files include a source mask distinguishing predictions, interpolation fallback and invalid pixels.

Release checks cover file completeness, syntax, entry-point availability, split membership and model loading. Agreement between current code, historical results and paper figures is outside this release-preparation scope. A complete data archive remains a separate task.

# Generate full-disk super-resolution magnetograms

This folder implements the full workflow from an HMI observation to a plotted reconstruction. The three existing paper figures and case settings are in [`examples/application/`](../examples/application/README.md). Their large input/result arrays are kept outside the current GitHub tree; six pretrained models remain in [`pre-trained/`](../pre-trained).

Install once from the repository root with `python -m pip install -e ".[solar]"`. The folder's Python module name is `full_disk`.

## 1. Download an observation

```bash
python -m full_disk.download --case examples/application/solar-maximum/case.json --email YOUR_REGISTERED_JSOC_EMAIL --output outputs/solar-maximum/input --with-sp
```

This selects the exact HMI `hmi.M_720s` record in the case file (record times are TAI), saves `hmi.fits`, and optionally downloads `sp.fits` for comparison. SP is not required by the reconstruction model. The download log records the retrieved file hashes. If the inputs are already available locally, skip this step and supply their paths below.

## 2. Reconstruct the full disk

```bash
python -m full_disk.generate --hmi outputs/solar-maximum/input/hmi.fits --checkpoint pre-trained/pm_ltew_cc.pth --output outputs/solar-maximum/result --device cuda
```

Use `--device cpu` when needed. A full disk involves up to 400 patch predictions and requires substantially more time and memory than a test patch. `--chunk-size` controls query memory; feature maps and the full assembled arrays still require additional memory.

The workflow follows the retained research scripts:

1. Reverse the HMI columns and derive projected radius from the original HMI WCS.
2. Centre a non-overlapping 200 × 200 patch grid. A 4096 × 4096 input loses 48 pixels on each side, leaving 20 × 20 tiles.
3. Predict each finite patch at 312 × 336 resolution, with HMI and radius divided by 200 as in patch inference.
4. Assemble the tiles into a 6240 × 6720 `SRfield` in G. Patches containing non-finite HMI pixels are not passed to the model.
5. Fill unpredicted regions using the original cubic interpolation convention, propagating an interpolated invalid mask. These values form `SRFfield` and are explicitly marked as interpolation fallback.

Outputs:

- `reconstruction.npz`: `SRfield` (float32), `SRFfield` (float64) and `source_mask` (uint8: 0 invalid, 1 model, 2 interpolation).
- `manifest.json`: model/input checksums, crop, scale, orientation, coordinate mapping and per-patch status.
- `input-header.txt`: original input FITS header for coordinate provenance.

The mask describes newly generated outputs; it is not a recovered mask for historical paper results. Model errors stop the command rather than silently substituting interpolation. Pixel values are in G; coordinate provenance is supplied separately instead of assigning an unverified output WCS.

## 3. Plot the result

```bash
python -m full_disk.plot --hmi outputs/solar-maximum/input/hmi.fits --result outputs/solar-maximum/result --case examples/application/solar-maximum/case.json --output outputs/solar-maximum/comparison.png --sp outputs/solar-maximum/input/sp.fits
```

This generates HMI/reconstruction full-disk and zoom panels; optional SP input produces a separate comparison-map PNG. Zoom boxes and color limits are supplied in the case files. The plotting command makes an inspection figure; the original manuscript composites remain in `examples/application`.

Replace `solar-maximum` with `solar-minimum` or `quiet-sun` for the other cases. A new HMI observation can be supplied directly to `generate`; omit `--case` in `plot` to use the default central zoom region.

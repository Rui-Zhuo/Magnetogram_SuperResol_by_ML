# Data access and provenance

## Available in this repository

- Exact 10,115-ID split: 6,069 train / 2,023 validation / 2,023 test.
- Three paired test observations, copied without changing their scientific arrays.
- Original test membership, AR-like labels and heliocentric angles for all 2,023 test samples.
- Six model checkpoints, exported without optimizer state. SHA-256 checksums identify both source files and exports.
- Three existing research predictions and their figures, with source filenames.
- Three full-disk paper figures with observation metadata, plus the complete [generation workflow](../full-disk/README.md).

The full paired dataset (~15 GB in its original NPZ storage) and complete raw FITS archive are **not** bundled. Large full-disk FITS/NPZ products are also excluded from the current tree; previews and source identifiers are included. Full-data archival is a separate task; no permanent archive URL or DOI is available in this release. A generation recipe is useful but is not equivalent to making the actual training and derived datasets downloadable.

## Raw sources

1. HMI LOS magnetograms: JSOC series `hmi.M_720s`, segment `magnetogram`, 720-second temporal averaging. See [JSOC](http://jsoc.stanford.edu/) and [SunPy's JSOC access documentation](https://docs.sunpy.org/en/stable/tutorial/acquiring_data/jsoc.html). Users supply their own registered JSOC email.
2. SP L2 and L2.1: [HAO/SP](https://www2.hao.ucar.edu/csac/csac-instruments/sp), mirrored by DARTS under `https://data.darts.isas.jaxa.jp/pub/hinode/sot/level2hao/` and `https://data.darts.isas.jaxa.jp/pub/hinode/sot/level2.1hao/`. The downloader expands each observation ID into year/month/day and the `SP3D` directory. L2 contains slit positions; L2.1 HDU 4 supplies the target array used by the original code.
3. Updated pointing/affine metadata: [dfouhey/SOTSPPointing](https://github.com/dfouhey/SOTSPPointing), accompanying Fouhey et al. (2023), ApJS 264, 49 and subsequent pointing updates. Follow its archive link and preserve the upstream version and checksum. The coarse pipeline requires FITS header keys `WARP00`…`WARP12` and `BNDMINX/Y`, `BNDMAXX/Y`.

The included IDs identify the historical sample selection. The release does not reimplement the original human/quality selection from every SP observation over the solar cycle. Downloads can fail because of remote availability; errors are written to a machine-readable report and cause a nonzero exit status. Raw data are not fetched during package import.

## Versioned processing

The coarse alignment follows `coalign.py` in the research tree. The new command-line interface reads L2 FITS directly instead of requiring an intermediate reduced FITS copy. WCS is evaluated only inside the crop; reversing both HMI axes is retained. The saved NPZ schema remains `HMIfield_crop`, `HMIfield_coalign`, `SPfield_coalign`, `Txy_crop`.

Fine registration is copied from `correct_coalign_loop.py`: local FFT phase correlation, parabolic subpixel peak fitting, spline interpolation of displacement, and cubic resampling. It does not use an external FLCT library. The local registration can change flux and artificially increase agreement; displacement statistics do not quantify causal calibration bias. HMI-grid centring, final dimensions and interpolation conventions are preserved. Removed `scipy.interpolate.interp2d` is replaced by its regular-grid `RectBivariateSpline` equivalent.

Inputs containing NaN or infinity are excluded with explicit errors. This can expose cases previously skipped by the research scripts. No missing file, failed pair or invalid field is silently replaced by zeros in the final paired dataset. The original coarse affine diagnostic used zero-filled non-finite HMI values; the final HMI crop retains its actual values and is checked before dataset generation.

## Separate archive still needed

For a complete data release, deposit the exact paired NPZ files, their checksums, the split and metadata, the HMI matching log, preprocessing/version provenance, and the full-disk products in a persistent repository. Include a data dictionary and the exact code commit. Full-disk files should carry WCS, units, timestamps, checkpoint ID, and a mask distinguishing learned output, interpolation fallback and invalid/off-limb pixels. Do not claim a mask for an old product unless its original patch provenance can be recovered.

The [AAS data guide](https://journals.aas.org/data-guide/) discusses repositories and availability. This code release is one part of that transparency work, not a replacement for the separately planned complete data archive.

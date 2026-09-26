# Full-disk applications

Three existing applications from the manuscript, reconstructed with **PM-LTEW-CC**:

| Case | Paper figure | Preview |
|---|---|---|
| [Solar maximum](solar-maximum) | Figure 6, main text | [Figure](solar-maximum/paper-figure.png) |
| [Solar minimum](solar-minimum) | Figure 7, main text | [Figure](solar-minimum/paper-figure.png) |
| [Quiet Sun](quiet-sun) | Figure 9, appendix | [Figure](quiet-sun/paper-figure.png) |

Each folder contains `hmi.fits`, `sp.fits`, `paper-figure.png`, a provenance/checksum `manifest.json`, and `results/part-*.npz`. Figures are extracted from the supplied manuscript. Input FITS files and result values are preserved from the research outputs.

The result arrays have shape **6240 × 6720**, in **G**:

- `SRfield` (`float32`): assembled reconstruction before final filling.
- `SRFfield` (`float64`): existing filled reconstruction used for display.

Each NPZ stores a consecutive row slab of both arrays. Lossless compression and splitting keep individual files below GitHub's file-size limit; the three cases together occupy about 850 MB including inputs. Row ranges, data types and SHA-256 checksums are recorded in each manifest.

## Read an array

From the repository root:

```python
from magnetosr.full_disk import load_result

field = load_result('full-disk/solar-maximum', key='SRFfield')
```

To reassemble both arrays into a single NPZ:

```bash
python -m magnetosr.full_disk --case full-disk/solar-maximum --output outputs/solar-maximum.npz
```

## Plot downloaded or reconstructed magnetograms

```bash
python -m pip install -e ".[solar]"
python -m magnetosr.plot_magnetogram --input full-disk/solar-maximum/hmi.fits --hdu 1 --output outputs/hmi.png
python -m magnetosr.plot_magnetogram --input full-disk/solar-maximum/sp.fits --hdu 4 --output outputs/sp.png
python -m magnetosr.plot_magnetogram --input full-disk/solar-maximum --key SRFfield --output outputs/reconstruction.png
```

The plotting command displays the selected array in pixel coordinates. The paper's composite figures are supplied separately. Input FITS headers retain observing times and WCS. Original reconstruction NPZ files have no WCS or separate provenance mask for filled/off-limb pixels; the manifest records the available crop, scale and display-orientation information. These examples do not replace the separately planned complete data archive.

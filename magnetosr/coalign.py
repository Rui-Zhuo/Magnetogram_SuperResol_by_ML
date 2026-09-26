"""Coarse alignment using published SP pointing/affine FITS metadata."""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.ndimage import affine_transform


def coarse_pair(pointing, spl2, spl21, hmi_file, output):
    from astropy.io import fits
    import astropy.units as u
    import sunpy.map
    with fits.open(pointing) as update, fits.open(spl2) as level2, fits.open(spl21) as level21:
        header = update[0].header
        slit = np.asarray(level2[41].data[0][0], dtype=float)
        locations = (slit-slit[0])/np.median(np.diff(slit))
        sp = np.array(level21[4].data, dtype=float)
        expanded_width = int(locations.max())+1
        affine_xy = np.array([[header[f'WARP{i}{j}'] for j in range(3)] for i in range(2)] + [[0, 0, 1]])
        affine_yx = affine_xy[np.ix_([1, 0, 2], [1, 0, 2])]
        bounds = [int(header[k]) for k in ['BNDMINY', 'BNDMAXY', 'BNDMINX', 'BNDMAXX']]
    hmi = sunpy.map.Map(hmi_file)
    field = hmi.data[::-1, ::-1]
    y0, y1, x0, x1 = bounds
    if not (0 <= y0 < y1 <= field.shape[0] and 0 <= x0 < x1 <= field.shape[1]):
        raise ValueError('Pointing crop bounds lie outside the HMI map')
    # Compute only the crop WCS, preserving the original 180-degree reversal.
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sky = hmi.pixel_to_world((field.shape[1]-1-xx)*u.pix, (field.shape[0]-1-yy)*u.pix)
    radius = np.hypot(sky.Tx.arcsec, sky.Ty.arcsec)
    warped = affine_transform(np.nan_to_num(field, nan=0.), affine_yx,
                              output_shape=(sp.shape[0], expanded_width), order=1)
    hmi_at_sp = warped[:, locations.astype(int)]
    np.savez_compressed(output, HMIfield_crop=field[y0:y1, x0:x1],
                        HMIfield_coalign=hmi_at_sp, SPfield_coalign=sp, Txy_crop=radius)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['pointing', 'sp-l2', 'sp-l21', 'hmi', 'pairs', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    failures = []
    for row in csv.DictReader(a.pairs.open()):
        sample = row['sample']
        try:
            coarse_pair(a.pointing/(sample+'.fits'), a.sp_l2/(sample+'.fits'),
                        a.sp_l21/(sample+'.fits'), a.hmi/row['hmi_file'], a.output/(sample+'.npz'))
            print(sample, flush=True)
        except Exception as e: failures.append((sample, str(e)))
    if failures:
        import json
        (a.output/'failures.json').write_text(json.dumps(failures, indent=2))
        raise SystemExit(f'{len(failures)} pairs failed; see failures.json')


if __name__ == '__main__': main()

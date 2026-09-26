"""Reconstruct a full HMI disk by patch inference, assembly and background filling."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from inference.predict import load_model, predict
from scipy.interpolate import RectBivariateSpline


PATCH = 200
TARGET = (312, 336)


def centered_grid(shape):
    """Even patch counts centred on the image; 4096 yields a 48-pixel border."""
    counts = tuple((size // (2 * PATCH)) * 2 for size in shape)
    if min(counts) == 0:
        raise ValueError('HMI image must contain at least two patches along each axis')
    offsets = tuple(size // 2 - count * PATCH // 2 for size, count in zip(shape, counts))
    return counts, offsets


def interpolated_background(field, shape):
    """Original cubic fill, with linearly resampled invalid-pixel mask."""
    h, w = field.shape
    y, x = np.linspace(0, h, h), np.linspace(0, w, w)
    yy, xx = np.linspace(0, h, shape[0]), np.linspace(0, w, shape[1])
    invalid = ~np.isfinite(field)
    clean = np.where(invalid, 0., field)
    filled = RectBivariateSpline(y, x, clean)(yy, xx)
    mask = RectBivariateSpline(y, x, invalid.astype(float), kx=1, ky=1)(yy, xx)
    filled[mask > .5] = np.nan
    return filled


def reconstruct(field, radius_at, model, chunk_size=8192):
    """Process non-overlapping patches; radius_at returns arcsec for a source tile."""
    counts, offsets = centered_grid(field.shape)
    rows, cols = counts
    y0, x0 = offsets
    shape = (rows * TARGET[0], cols * TARGET[1])
    sr = np.full(shape, np.nan, dtype=np.float32)
    records = []
    for row in range(rows):
        for col in range(cols):
            y, x = y0 + row * PATCH, x0 + col * PATCH
            patch = field[y:y+PATCH, x:x+PATCH]
            record = dict(row=row, column=col, y_start=y, x_start=x)
            if not np.isfinite(patch).all():
                record['status'] = 'interpolation: non-finite HMI patch'
            else:
                radius = radius_at(y, x, PATCH)
                if radius.shape != patch.shape or not np.isfinite(radius).all():
                    raise ValueError(f'Invalid projected-radius coordinates at row {row}, column {col}')
                tile = predict(model, patch, radius, TARGET, chunk_size)
                if tile.shape != TARGET or not np.isfinite(tile).all():
                    raise ValueError(f'Invalid prediction at row {row}, column {col}')
                yy, xx = row * TARGET[0], col * TARGET[1]
                sr[yy:yy+TARGET[0], xx:xx+TARGET[1]] = tile
                record['status'] = 'model'
            records.append(record)
            print(f'Patch {len(records)}/{rows*cols}: {record["status"]}', flush=True)
    cropped = field[y0:y0+rows*PATCH, x0:x0+cols*PATCH]
    filled = interpolated_background(cropped, shape)
    learned = np.isfinite(sr)
    filled[learned] = sr[learned]
    source = np.zeros(shape, dtype=np.uint8)
    source[np.isfinite(filled)] = 2
    source[learned] = 1
    return dict(SRfield=sr, SRFfield=filled, source_mask=source), records, offsets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hmi', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('pre-trained/pm_ltew_cc.pth'))
    p.add_argument('--output', type=Path, required=True, help='Directory for generated NPZ and metadata')
    p.add_argument('--device', default='cpu')
    p.add_argument('--chunk-size', type=int, default=8192)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.chunk_size <= 0 or a.threads <= 0:
        p.error('chunk-size and threads must be positive')
    import astropy.units as u
    import sunpy.map
    import torch
    torch.set_num_threads(a.threads)
    hmi = sunpy.map.Map(a.hmi)
    if hmi.data.ndim != 2:
        p.error('Expected a two-dimensional HMI magnetogram')
    # Match the original full-disk preparation: reverse columns only.
    field = np.asarray(hmi.data[:, ::-1], dtype=np.float32)

    def radius_at(y, x, size):
        yy, xx = np.mgrid[y:y+size, x:x+size]
        sky = hmi.pixel_to_world((field.shape[1]-1-xx)*u.pix, yy*u.pix)
        return np.hypot(sky.Tx.arcsec, sky.Ty.arcsec).astype(np.float32)

    model = load_model(a.checkpoint, a.device)
    arrays, records, offsets = reconstruct(field, radius_at, model, a.chunk_size)
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output/'reconstruction.npz', **arrays)
    # The input header plus this explicit pixel transform defines output geometry.
    hmi.fits_header.totextfile(a.output/'input-header.txt', overwrite=True)
    metadata = dict(
        checkpoint=a.checkpoint.name,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(a.hmi.read_bytes()).hexdigest(),
        input_shape=list(field.shape), output_shape=list(arrays['SRfield'].shape),
        unit='G', radius_unit='arcsec', patch_shape=[PATCH, PATCH], target_patch_shape=list(TARGET),
        crop_y=int(offsets[0]), crop_x=int(offsets[1]), column_reversal=True,
        scale_y=TARGET[0]/PATCH, scale_x=TARGET[1]/PATCH,
        coordinate_mapping='Within tile (row,col), output index q corresponds to input-centre coordinate: crop + tile*200 + (q+0.5)*200/target_size - 0.5; reverse the resulting x about input_width-1 for input WCS. The interpolation-only background uses endpoint-aligned sampling.',
        source_mask={'0':'invalid', '1':'model prediction', '2':'interpolation fallback'},
        interpolation='Cubic field interpolation; linear invalid mask with threshold 0.5',
        patch_records=records)
    (a.output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(a.output/'reconstruction.npz', flush=True)


if __name__ == '__main__':
    main()
